import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams.update({
    'font.size': 18,
    'lines.linewidth': 3}
)

control_results = {
    1e-1 : np.array([]) ,
    1e-2 : np.array([]),
    1e-3 : np.array([]),
    1e-4 : np.array([]),
    1e-5 : np.array([]),
    1e-6 : np.array([]),
}


def main(results, label):
    best_train_iou = []
    best_val_iou = []

    best_training_loss = []
    best_val_loss = []

    lr_arr = []

    for key, value in sorted(results.items()):

        validation_iou_arr = value[:, 4]
        train_iou_arr = value[:, 2]
        validation_loss_arr = value[:, 3]
        train_loss_arr = value[:, 1]

        lr_arr.append(key)
        best_train_iou.append(np.max(train_iou_arr))
        best_val_iou.append(np.max(validation_iou_arr))
        best_training_loss.append(np.min(train_loss_arr))
        best_val_loss.append(np.min(validation_loss_arr))

    # Plot best Iou vs Tau
    plt.figure("IoU")
    plt.plot(lr_arr, best_train_iou, label='train_' + label, marker='x', markersize=10, markeredgewidth=3)
    plt.plot(lr_arr, best_val_iou, label='val_' + label, marker='x', markersize=10, markeredgewidth=3)
    plt.xlabel("learning rate")
    plt.ylabel("IoU")
    plt.title("IoU vs Learning Rate")
    plt.legend()
    plt.grid()

    # Plot lowest loss vs Tau
    plt.figure("Loss")
    plt.plot(lr_arr, best_training_loss, label='train_' + label, marker='x', markersize=10, markeredgewidth=3)
    plt.plot(lr_arr, best_val_loss, label='val_' + label, marker='x', markersize=10, markeredgewidth=3)
    plt.xlabel("learning rate")
    plt.ylabel("Loss")
    plt.title("IoU vs Learning Rate")
    plt.legend()
    plt.grid()


if __name__ == "__main__":
    plt.ion()

    # main(tau_results_old, 'old')
    main(control_results, 'control')

    # ----------------------------------------------------------------------
    import pdb
    pdb.set_trace()