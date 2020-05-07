import torch.nn as nn
import torchvision

from .new_piech_models import ClassifierHead


class ControlMatchParametersLayer(nn.Module):

    def __init__(self, edge_out_ch=64, lateral_e_size=7, lateral_i_size=7):
        super(ControlMatchParametersLayer, self).__init__()

        self.lateral_e_size = lateral_e_size
        self.lateral_i_size = lateral_i_size
        self.edge_out_ch = edge_out_ch

        self.lateral_e = nn.Conv2d(
            in_channels=edge_out_ch,
            out_channels=edge_out_ch,
            kernel_size=self.lateral_e_size,
            stride=1,
            padding=(self.lateral_e_size - 1) // 2,
            bias=False
        )

        self.control_bn1 = nn.BatchNorm2d(num_features=edge_out_ch)
        self.control_dp1 = nn.Dropout(p=0.3)

        self.lateral_i = nn.Conv2d(
            in_channels=edge_out_ch,
            out_channels=edge_out_ch,
            kernel_size=self.lateral_i_size,
            stride=1,
            padding=(self.lateral_i_size - 1) // 2,
            bias=False
        )

        self.control_bn2 = nn.BatchNorm2d(num_features=edge_out_ch)
        self.control_dp2 = nn.Dropout(p=0.3)

    def forward(self, ff):
        ff = self.lateral_e(ff)
        ff = self.control_bn1(ff)
        ff = nn.functional.relu(ff)
        ff = self.control_dp1(ff)

        ff = self.lateral_i(ff)
        ff = self.control_bn2(ff)
        ff = nn.functional.relu(ff)
        ff = self.control_dp2(ff)

        return ff


class ControlMatchIterationsLayer(nn.Module):
    def __init__(self, edge_out_ch=64, n_iters=5, lateral_e_size=7, lateral_i_size=7):
        super(ControlMatchIterationsLayer, self).__init__()

        self.lateral_e_size = lateral_e_size
        self.lateral_i_size = lateral_i_size
        self.edge_out_ch = edge_out_ch
        self.n_iters = n_iters

        self.lateral_e = nn.Conv2d(
            in_channels=edge_out_ch,
            out_channels=edge_out_ch,
            kernel_size=self.lateral_e_size,
            stride=1,
            padding=(self.lateral_e_size - 1) // 2,
            bias=False
        )

        self.control_bn1 = nn.BatchNorm2d(num_features=edge_out_ch)
        self.control_dp1 = nn.Dropout(p=0.3)

        self.lateral_i = nn.Conv2d(
            in_channels=edge_out_ch,
            out_channels=edge_out_ch,
            kernel_size=self.lateral_i_size,
            stride=1,
            padding=(self.lateral_i_size - 1) // 2,
            bias=False
        )

        self.control_bn2 = nn.BatchNorm2d(num_features=edge_out_ch)
        self.control_dp2 = nn.Dropout(p=0.3)

    def forward(self, ff):

        for i in range(self.n_iters):
            ff = self.lateral_e(ff)
            ff = self.control_bn1(ff)
            ff = nn.functional.relu(ff)
            ff = self.control_dp1(ff)

            ff = self.lateral_i(ff)
            ff = self.control_bn2(ff)
            ff = nn.functional.relu(ff)
            ff = self.control_dp2(ff)

        return ff


class ControlMatchParametersModel(nn.Module):
    """
    Full Model With Control Layer that matches the number of parameters (Number of convolutional
    layers and their spatial extent.
    """

    def __init__(self, lateral_e_size=7, lateral_i_size=7):
        super(ControlMatchParametersModel, self).__init__()

        # # First Convolutional Layer of Alexnet
        # self.edge_extract = torchvision.models.alexnet(pretrained=True).features[0]
        # self.edge_extract.weight.requires_grad = False
        # self.edge_extract.bias.requires_grad = False

        self.edge_extract = nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2, bias=False)
        alexnet_kernel = torchvision.models.alexnet(pretrained=True).features[0]
        self.edge_extract.weight.data = alexnet_kernel.weight.data
        self.edge_extract.requires_grad = False

        self.num_edge_extract_chan = self.edge_extract.weight.shape[0]

        self.bn1 = nn.BatchNorm2d(num_features=self.num_edge_extract_chan)

        # Current Subtractive Layer
        self.contour_integration_layer = ControlMatchParametersLayer(
            edge_out_ch=self.num_edge_extract_chan,  # number of channels of edge extract layer
            lateral_e_size=lateral_e_size,
            lateral_i_size=lateral_i_size
        )

        # Classifier
        self.classifier = ClassifierHead(self.num_edge_extract_chan)

    def forward(self, in_img):
        # Edge Extraction
        x = self.edge_extract(in_img)
        x = self.bn1(x)
        x = nn.functional.relu(x)

        x = self.contour_integration_layer(x)

        x = self.classifier(x)

        return x
