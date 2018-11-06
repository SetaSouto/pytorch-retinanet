"""Training module script."""

import argparse
import collections
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import coco_eval
import csv_eval
import model
from dataloader import (AspectRatioBasedSampler, Augmenter, CocoDataset,
                        CSVDataset, Normalizer, Resizer, collater)
from torchvision import transforms

assert torch.__version__.split('.')[1] == '4'

print('CUDA available: {}'.format(torch.cuda.is_available()))


def main(args=None):
    """Main function to execute during training."""

    parser = argparse.ArgumentParser(description='Simple training script for training a RetinaNet network.')
    parser.add_argument('--dataset', help='Dataset type, must be one of csv or coco.')
    parser.add_argument('--coco_path', help='Path to COCO directory')
    parser.add_argument('--csv_train', help='Path to file containing training annotations (see readme)')
    parser.add_argument('--csv_classes', help='Path to file containing class list (see readme)')
    parser.add_argument('--csv_val', help='Path to file containing validation annotations (optional, see readme) ')
    parser.add_argument('--depth', help='Resnet depth, must be one of 18, 34, 50, 101, 152', type=int, default=50)
    parser.add_argument('--epochs', help='Number of epochs', type=int, default=100)

    parser = parser.parse_args(args)

    # Create the data loaders
    if parser.dataset == 'coco':
        if parser.coco_path is None:
            raise ValueError('Must provide --coco_path when training on COCO,')

        dataset_train = CocoDataset(parser.coco_path, set_name='train2014',
                                    transform=transforms.Compose([Normalizer(), Augmenter(), Resizer()]))
        dataset_val = CocoDataset(parser.coco_path, set_name='val2014',
                                  transform=transforms.Compose([Normalizer(), Resizer()]))

    elif parser.dataset == 'csv':
        if parser.csv_train is None:
            raise ValueError('Must provide --csv_train when training on CSV')
        if parser.csv_classes is None:
            raise ValueError('Must provide --csv_classes when training on CSV')

        dataset_train = CSVDataset(train_file=parser.csv_train, class_list=parser.csv_classes,
                                   transform=transforms.Compose([Normalizer(), Augmenter(), Resizer()]))

        if parser.csv_val is None:
            dataset_val = None
            print('No validation annotations provided.')
        else:
            dataset_val = CSVDataset(train_file=parser.csv_val, class_list=parser.csv_classes,
                                     transform=transforms.Compose([Normalizer(), Resizer()]))

    else:
        raise ValueError(
            'Dataset type not understood (must be csv or coco), exiting.')

    sampler = AspectRatioBasedSampler(dataset_train, batch_size=1, drop_last=False)
    dataloader_train = DataLoader(dataset_train, num_workers=1, collate_fn=collater, batch_sampler=sampler)

    # What do these lines do? Can we we delete them?
    if dataset_val is not None:
        sampler_val = AspectRatioBasedSampler(dataset_val, batch_size=1, drop_last=False)
        dataloader_val = DataLoader(dataset_val, num_workers=3, collate_fn=collater, batch_sampler=sampler_val)

    # Create the model
    if parser.depth == 18:
        retinanet = model.resnet18(num_classes=dataset_train.num_classes(), pretrained=True)
    elif parser.depth == 34:
        retinanet = model.resnet34(num_classes=dataset_train.num_classes(), pretrained=True)
    elif parser.depth == 50:
        retinanet = model.resnet50(num_classes=dataset_train.num_classes(), pretrained=True)
    elif parser.depth == 101:
        retinanet = model.resnet101(num_classes=dataset_train.num_classes(), pretrained=True)
    elif parser.depth == 152:
        retinanet = model.resnet152(num_classes=dataset_train.num_classes(), pretrained=True)
    else:
        raise ValueError('Unsupported model depth, must be one of 18, 34, 50, 101, 152')

    retinanet = retinanet.cuda()
    retinanet = torch.nn.DataParallel(retinanet).cuda()
    retinanet.training = True
    optimizer = optim.Adam(retinanet.parameters(), lr=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True)
    loss_hist = collections.deque(maxlen=500)
    retinanet.train()
    retinanet.module.freeze_bn()

    print('Num training images: {}'.format(len(dataset_train)))

    training_start_time = time.time()
    logs_file = open('logs_{}_resnet_{}.txt'.format(parser.dataset, parser.depth), 'w')

    for epoch_num in range(parser.epochs):
        retinanet.train()
        retinanet.module.freeze_bn()

        epoch_loss = []
        epoch_start_time = time.time()

        try:
            for iter_num, data in enumerate(dataloader_train):
                iteration_start_time = time.time()

                try:
                    optimizer.zero_grad()

                    classification_loss, regression_loss = retinanet([data['img'].cuda().float(), data['annot']])
                    classification_loss = classification_loss.mean()
                    regression_loss = regression_loss.mean()

                    loss = classification_loss + regression_loss

                    if loss == 0:
                        continue

                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(retinanet.parameters(), 0.1)
                    optimizer.step()
                    loss_hist.append(float(loss))
                    epoch_loss.append(float(loss))

                    # Logging
                    actual_time = time.time()
                    log = 'Epoch: {} | Iteration: {} | Classification loss: {:1.5f} | Regression loss: {:1.5f} |' \
                        'Running loss: {:1.5f} | Times: training: {:.3f}, epoch: {:.3f}, iteration: {:.3f}' \
                        .format(epoch_num, iter_num, float(classification_loss), float(regression_loss),
                                np.mean(loss_hist), actual_time - training_start_time,
                                actual_time - epoch_start_time, actual_time - iteration_start_time)
                    print(log)
                    logs_file.write(log + '\n')

                    if iter_num != 0 and iter_num % 999 == 0:
                        torch.save(retinanet.module,
                                   '{}_retinanet_unfinished_epoch_{}.pt'.format(parser.dataset, epoch_num))

                    # Free memory
                    del classification_loss
                    del regression_loss
                except Exception as exception:
                    print(exception)
                    continue
        except Exception as exception:
            # Exception for an error during loading the data, some images could be "truncated" and couldn't be loaded
            print(exception)
            continue

        if parser.dataset == 'coco':
            print('Evaluating dataset')
            coco_eval.evaluate_coco(dataset_val, retinanet)

        elif parser.dataset == 'csv' and parser.csv_val is not None:
            print('Evaluating dataset')
            mAP = csv_eval.evaluate(dataset_val, retinanet)

        scheduler.step(np.mean(epoch_loss))
        torch.save(retinanet.module, '{}_retinanet_epoch_{}.pt'.format(parser.dataset, epoch_num))

    retinanet.eval()
    torch.save(retinanet, 'model_final_{}.pt'.format(epoch_num))
    logs_file.close()


if __name__ == '__main__':
    main()
