# ---------------------------------------------------------------------------------------
#  View the internal parameters of a trained contour integration model
#
# NOTE: The following temp code needs to be added to models/new_piech_models.py
#
#   >> # # Debug
#   >> # idx = ff.argmax()  # This is the index in the flattened array
#   import matplotlib.pyplot as plt
#   import numpy as np
#   f, ax_arr = plt.subplots(2, 5, figsize=(15, 5))
#   ax_arr[0, 0].set_ylabel("f(x)")
#   ax_arr[0, 1].set_ylabel("f(y)")
#   >> ...
#
#   >> for i in range(self.n_iters):
#   >> ..
#   >> f_x = nn.functional.relu(x)
#   >> f_y = nn.functional.relu(y)
#   disp_x = f_x.detach().cpu().numpy()
#   disp_x = np.squeeze(disp_x)
#   disp_x = np.sum(disp_x, axis=0)  # some across channels
#
#   disp_y = f_y.detach().cpu().numpy()
#   disp_y = np.squeeze(disp_y)
#   disp_y = np.sum(disp_y, axis=0)  # some across channels
#
#   im_x = ax_arr[0, i].imshow(disp_x)
#   plt.colorbar(im_x, ax=ax_arr[0, i], orientation='horizontal')
#   im_y = ax_arr[1, i].imshow(disp_y)
#   plt.colorbar(im_y, ax=ax_arr[1, i], orientation='horizontal')
# ---------------------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt

import torch

import models.new_piech_models as new_piech_models


def sigmoid(x):
    return 1. / (1 + np.exp(-x))


# ---------------------------------------------------------------------------------------
if __name__ == "__main__":
    plt.ion()

    cont_int_layer = new_piech_models.CurrentSubtractInhibitLayer(
        lateral_e_size=15, lateral_i_size=15, n_iters=5)
    saved_model = \
        "./results/new_model_resnet_based/" \
        "ContourIntegrationResnet50_CurrentSubtractInhibitLayer_run_1_20200924_183734" \
        "/best_accuracy.pth"
    # saved_model = \
    #     '/home/salman/Desktop/' \
    #     'ContourIntegrationResnet50_CurrentSubtractInhibitLayer_20201205_102403_postive_weights_by_loss_fixed_jxy_jyx_sigma_b/' \
    #     'best_accuracy.pth'

    net = new_piech_models.ContourIntegrationResnet50(cont_int_layer)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = net.to(device)
    net.load_state_dict(torch.load(saved_model, map_location=device))

    print("Model Loaded")
    # -----------------------------------------------------------------------------------
    # [1] Jxy
    j_xy = net.contour_integration_layer.j_xy.data
    plt.figure()
    plt.plot(j_xy)
    plt.title('J_xy')
    plt.xlabel("Channel")
    plt.ylabel('J_xy')

    # [2] Jyx
    j_yx = net.contour_integration_layer.j_yx.data
    plt.figure()
    plt.plot(j_yx)
    plt.title('J_yx')
    plt.xlabel("Channel")
    plt.ylabel('J_yx')

    # [3] sigma(a)
    a = net.contour_integration_layer.a.data
    sigma_a = sigmoid(a)
    plt.figure()
    plt.plot(sigma_a)
    plt.title('sigma(a)')
    plt.xlabel("Channel")
    plt.ylabel('sigma(a)')
    plt.ylim([0, 1])

    # [4] sigma(b)
    b = net.contour_integration_layer.b.data
    sigma_b = sigmoid(b)
    plt.figure()
    plt.plot(sigma_b)
    plt.title('sigma(b)')
    plt.xlabel("Channel")
    plt.ylabel('sigma(b)')
    plt.ylim([0, 1])

    # [5] I_bias
    i_bias = net.contour_integration_layer.i_bias.data
    plt.figure()
    plt.plot(i_bias)
    plt.title('i_bias')
    plt.xlabel("Channel")
    plt.ylabel('i_bias')

    # [6] I_bias
    e_bias = net.contour_integration_layer.e_bias.data
    plt.figure()
    plt.plot(e_bias)
    plt.title('e_bias')
    plt.xlabel("Channel")
    plt.ylabel('e_bias')

    # -----------------------------------------------------------------------------------
    # Single Plot of all parameters for each Channel
    plt.figure()
    plt.plot(j_xy, label='J_xy', marker='x')
    plt.plot(j_yx, label='J_yx', marker='o')
    plt.plot(sigma_a, label='sigma(a)', marker='s')
    plt.plot(sigma_b, label='sigma(b)', marker='d')
    plt.plot(e_bias, label='e_bias', marker='v')
    plt.plot(i_bias, label='i_bias', marker='^')
    plt.title('All parameters per channel')
    plt.xlabel("Channel")
    plt.legend()

    # -----------------------------------------------------------------------------------
    print('End')
    import pdb
    pdb.set_trace()
