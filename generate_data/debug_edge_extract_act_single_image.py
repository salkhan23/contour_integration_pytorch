# ---------------------------------------------------------------------------------------
# This is a debug script for checking edge extract activations to a single file
#
# The image is normally loaded and all normalizations are done manually.
# Gives the same output as when pytorch does it through its data loaders
#
# ---------------------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt

from torchvision.models import alexnet
import torch


edge_extract_act = 0


def edge_extract_cb(self, layer_in, layer_out):
    global edge_extract_act
    edge_extract_act = torch.relu(layer_out).cpu().detach().numpy()


if __name__ == "__main__":
    random_seed = 20
    plt.ion()
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    fragment_size = (7, 7)
    full_tile_size = np.array([14, 14])
    image_size = np.array([256, 256, 3])

    # Model
    # --------------------------------------
    # Alexnet
    net = alexnet(pretrained=True)
    net.features[0].register_forward_hook(edge_extract_cb)

    # from models.new_piech_models import ContourIntegrationCSI
    # net = ContourIntegrationCSI(lateral_e_size=15, lateral_i_size=15, n_iters=5)
    # saved_model = './results/analyze_lr_rate_alexnet_bias/lr_3e-05/ContourIntegrationCSI_20191224_050603/' \
    #               'best_accuracy.pth'
    # net.load_state_dict(torch.load(saved_model))
    # net.edge_extract.register_forward_hook(edge_extract_cb)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = net.to(device)

    # Data
    # ---------------------------------------------------------
    image_file = './data/channel_wise_optimal_full14_frag7_centered/val/images/frag_3/clen_9/beta_0/alpha_0/' \
                 'clen_9_beta_0_alpha_0_0.png'

    img = plt.imread(image_file, format='PNG')
    img = img[:, :, 0:3]
    plt.imshow(img)
    plt.title("Input Image")

    mean = np.array([0.4208942, 0.4208942, 0.4208942])
    std = np.array([0.15286704, 0.15286704, 0.15286704])

    test_image = (img - img.min()) / (img.max() - img.min())  # [0,1] Pixel Range
    input_image = (test_image - mean) / std

    input_image = torch.from_numpy(np.transpose(input_image, axes=(2, 0, 1)))
    input_image = input_image.to(device)
    input_image = input_image.float()
    input_image = input_image.unsqueeze(dim=0)

    # Pass Through Model
    edge_extract_act = 0
    label_out = net(input_image)

    center_neuron_extract_out = \
        edge_extract_act[0, :, edge_extract_act.shape[2] // 2, edge_extract_act.shape[3] // 2]

    max_active_neuron = np.argmax(center_neuron_extract_out)
    string = "Edge Extract Activations of Center Neuron.\nMax Active Neuron {}. Value={:0.2f}.".format(
        max_active_neuron, center_neuron_extract_out[max_active_neuron])

    print(string)

    # Plot Center Neuron Activations
    plt.figure()
    plt.plot(center_neuron_extract_out)
    plt.title(string)
    plt.xlabel("Channel Index")
    plt.axvline(max_active_neuron, color='red')
    plt.grid()

    import pdb
    pdb.set_trace()
