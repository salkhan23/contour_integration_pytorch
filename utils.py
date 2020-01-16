import numpy as np
import torch


def normalize_image(img):
    """
    Rescale image to [0,1] range for display using matplotlib
    :param img:
    :return:
    """
    img_max = np.max(img)
    img_min = np.min(img)

    new_image = np.copy(img)
    new_image = (new_image - img_min) / (img_max - img_min)

    return new_image


# ---------------------------------------------------------------------------------------
# Performance Metrics
# ---------------------------------------------------------------------------------------
def intersection_over_union(outputs, targets):
    """
    Calculates Jaccard Distance.

    REF:  https://github.com/arturml/pytorch-unet

    :param outputs:
    :param targets:
    :return:
    """
    outputs = outputs.view(outputs.size(0), -1)
    targets = targets.view(targets.size(0), -1)
    intersection = (outputs * targets).sum(1)  # sum across images
    # -intersection to get rid of double values for correct preds
    union = (outputs + targets).sum(1) - intersection

    jac = (intersection + 0.001) / (union + 0.001)

    return jac.mean()


def class_balanced_cross_entropy(outputs, targets):
    """

    Dynamically finds the number of contours and non-egdes are dynamically scales their losses
    accordingly.

    NOTE: THIs function requires sigmoided outputs

    REF: originally proposed in
    [Xie, S., Tu, Z.:  Holistically-nested edge detection.
    In: Proceedings of the IEEE international conference on computer vision. (2015) 1395–1403]

    Used Reference:
    [Wang, Liang and Li -2018- DOOBNet: Deep Object Occlusion BoundaryDetection from an Image

    :param outputs:
    :param targets:
    :return:
    """

    n_total = targets.shape[0] * targets.shape[1] * targets.shape[2] * targets.shape[3]
    n_non_contours = torch.nonzero(targets).shape[0]
    n_contours = n_total - n_non_contours

    alpha = n_non_contours / n_total

    # print("Batch: Num Fragments={}, Num contour={}, num non-contour={}. alpha = {}".format(
    #      n_total, n_contours, n_non_contours, alpha))

    contour_loss = -targets * alpha * torch.log(outputs)
    non_contour_loss = (targets - 1) * (1 - alpha) * torch.log(1-outputs)

    loss = torch.sum(contour_loss + non_contour_loss)
    loss = loss / outputs.shape[0]  # batch size

    return loss
