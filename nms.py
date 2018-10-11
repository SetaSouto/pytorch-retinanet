"""Non-Maximum Suppression module."""
import numpy as np
import torch


def nms(detections, threshold):
    """Apply Non-Maximum Suppression over the detections.
    The detections must be a tensor with two dimensions: (number of detections, 5).
    Why 5? Because a detection has x1, y1, x2, y2 and score.

    Heavily inspired by Adrian Rosebrock at:
    https://www.pyimagesearch.com/2015/02/16/faster-non-maximum-suppression-python/

    Why not the version of GPU? Because I couldn't make it work in my GPU.

    Args:
        detections (torch.Tensor): A tensor with all the detections. The shape must be
            (number of detections, 5) with the score as the last value of the second
            dimension.
        threshold (float): The threshold for the IoU (intersection over union) to take
            two detections as detecting the same object.

    Returns:
        torch.Tensor: A tensor with the indexes of the detections to keep.
    """
    # If there aren't detections return empty
    if detections.shape[0] == 0:
        return torch.zeros((0))

    # Get the numpy version
    was_cuda = detections.is_cuda
    detections = detections.cpu().numpy()

    # Start the picked indexes list empty
    picked = []

    # Get the coordinates
    x1 = detections[:, 0]
    y1 = detections[:, 1]
    x2 = detections[:, 2]
    y2 = detections[:, 3]
    scores = detections[:, 4]

    # Compute the area of the bounding boxes
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)

    # Get the indexes of the detections sorted by score
    indexes = np.argsort(scores)

    while len(indexes) > 0:
        # Take the last index (highest score) and add it to the picked
        last = len(indexes) - 1
        actual = indexes[last]
        picked.append(actual)

        # We need to find the overlap of the bounding boxes with the actual picked bounding box

        # Find the largest (more to the bottom-right) (x,y) coordinates for the start
        # (top-left) of the bounding box
        xx1 = np.maximum(x1[actual], x1[indexes[:last]])
        yy1 = np.maximum(y1[actual], y1[indexes[:last]])
        # Find the smallest (more to the top-left) (x,y) coordinates for the end (bottom-right)
        # of the bounding box
        xx2 = np.minimum(x2[actual], x2[indexes[:last]])
        yy2 = np.minimum(y2[actual], y2[indexes[:last]])

        # Compute width and height to compute the intersection over union
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        intersection = (w * h)
        union = areas[actual] + areas[indexes[:last]] - intersection
        iou = intersection / union
        
        # Delete the last index and all that overlap is bigger than threshold
        indexes = np.delete(indexes, np.concatenate(([last], np.where(iou > threshold)[0])))

    # Return the picked indexes
    picked = torch.Tensor(picked).long()
    if was_cuda:
        picked = picked.cuda()

    return picked
