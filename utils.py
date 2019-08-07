import numpy as np


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
