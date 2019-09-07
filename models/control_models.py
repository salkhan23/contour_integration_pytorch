import torch.nn as nn
import torch.nn.functional as nn_functional
import torchvision

from .piech_models import ClassifierHead, ClassifierHeadOld


class CmMatchParameters(nn.Module):
    def __init__(self, edge_out_ch=64, lateral_e_size=7, lateral_i_size=7):
        """ Match the number of parameters """

        super(CmMatchParameters, self).__init__()

        # Technically this should be get layer predictions. But just to match the parameter
        # of the contour integration models use the same name.
        self.get_iterative_predictions = False

        # Parameters
        self.edge_out_ch = edge_out_ch
        self.lateral_e_size = lateral_e_size
        self.lateral_i_size = lateral_i_size

        # Layers
        # ------
        # First layer is AlexNet Edge Extraction (with out bias)
        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=self.edge_out_ch,
            kernel_size=11,
            stride=4,
            padding=2,
            bias=False
        )
        # use pre-trained Alexnet weights
        alexnet_edge_detect_kernels = torchvision.models.alexnet(pretrained=True).features[0]
        self.conv1.weight.data.copy_(alexnet_edge_detect_kernels.weight.data)

        # Additional Batch normalization Layer
        self.bn1 = nn.BatchNorm2d(num_features=64)

        self.control_conv1 = nn.Conv2d(
            in_channels=edge_out_ch,
            out_channels=edge_out_ch,
            kernel_size=self.lateral_e_size,
            stride=1,
            padding=(self.lateral_e_size - 1) // 2,
            bias=False
        )

        self.control_bn1 = nn.BatchNorm2d(num_features=edge_out_ch)
        self.control_dp1 = nn.Dropout(p=0.3)

        self.control_conv2 = nn.Conv2d(
            in_channels=edge_out_ch,
            out_channels=edge_out_ch,
            kernel_size=self.lateral_i_size,
            stride=1,
            padding=(self.lateral_i_size - 1) // 2,
            bias=False
        )

        self.control_bn2 = nn.BatchNorm2d(num_features=edge_out_ch)
        self.control_dp2 = nn.Dropout(p=0.3)

        self.post = ClassifierHead(edge_out_ch)

    def forward(self, in_img):

        # Edge Extraction
        ff = self.conv1(in_img)
        ff = self.bn1(ff)
        ff = nn.functional.relu(ff)

        # Control - Contour Integration equivalent
        out_layer_arr = []

        ff = self.control_conv1(ff)
        ff = self.control_bn1(ff)
        ff = nn.functional.relu(ff)
        if self.get_iterative_predictions:
            out_layer_arr.append(self.post(ff))
        ff = self.control_dp1(ff)

        ff = self.control_conv2(ff)
        ff = self.control_bn2(ff)
        ff = nn.functional.relu(ff)
        if self.get_iterative_predictions:
            out_layer_arr.append(self.post(ff))
        ff = self.control_dp2(ff)

        # Post processing
        # Post processing
        if self.get_iterative_predictions:
            out = self.post(ff), out_layer_arr
        else:
            out = self.post(ff)

        return out


class CmMatchIterations(nn.Module):
    def __init__(self, edge_out_ch=64, n_iters=5, lateral_e_size=7, lateral_i_size=7):
        """
        A Re-current model with matching number of recurrent iterations and parameters
        Parameters are matched by  using a two convolutional layers whose activities are subtracted
        before the batch normalization layer

        Role: Investigate the impact of the local circuit (which is not present in this model)
        """

        super(CmMatchIterations, self).__init__()

        # Technically this should be get layer predictions. But just to match the parameter
        # of the contour integration models use the same name.
        self.get_iterative_predictions = False

        # Parameters
        self.edge_out_ch = edge_out_ch
        self.n_iters = n_iters
        self.lateral_e_size = lateral_e_size
        self.lateral_i_size = lateral_i_size

        # Layers
        # ------
        # First layer is AlexNet Edge Extraction (with out bias)
        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=self.edge_out_ch,
            kernel_size=11,
            stride=4,
            padding=2,
            bias=False
        )

        # use pre-trained Alexnet weights
        alexnet_edge_detect_kernels = torchvision.models.alexnet(pretrained=True).features[0]
        self.conv1.weight.data.copy_(alexnet_edge_detect_kernels.weight.data)

        # Additional Batch normalization Layer
        self.bn1 = nn.BatchNorm2d(num_features=64)

        self.control_conv1 = nn.Conv2d(
            in_channels=edge_out_ch,
            out_channels=edge_out_ch,
            kernel_size=self.lateral_e_size,
            stride=1,
            padding=(self.lateral_e_size - 1) // 2,
            bias=False
        )
        self.control_conv2 = nn.Conv2d(
            in_channels=edge_out_ch,
            out_channels=edge_out_ch,
            kernel_size=self.lateral_i_size,
            stride=1,
            padding=(self.lateral_i_size - 1) // 2,
            bias=False
        )

        self.control_bn1 = nn.BatchNorm2d(num_features=edge_out_ch)
        self.control_dp1 = nn.Dropout(p=0.3)

        self.post = ClassifierHead(edge_out_ch)

    def forward(self, in_img):

        # Edge Extraction
        ff = self.conv1(in_img)
        ff = self.bn1(ff)
        ff = nn.functional.relu(ff)

        # Control - Contour Integration equivalent
        out_layer_arr = []

        for i in range(self.n_iters):
            ff_branch_a = self.control_conv1(ff)
            ff_branch_b = self.control_conv2(ff)

            ff = self.control_bn1(ff_branch_a - ff_branch_b)
            ff = nn.functional.relu(ff)

            if self.get_iterative_predictions:
                out_layer_arr.append(self.post(ff))

            ff = self.control_dp1(ff)

        # Post processing
        if self.get_iterative_predictions:
            out = self.post(ff), out_layer_arr
        else:
            out = self.post(ff)

        return out


class CmClassificationHeadOnly(nn.Module):
    def __init__(self, edge_out_ch=64, ):
        """ Simplest of control models, lateral layer is completely removed.
            THe purpose of this model is to ensure the classification head is not good enough to do
            the task on its own.
        """

        super(CmClassificationHeadOnly, self).__init__()

        self.get_iterative_predictions = False

        # Parameters
        self.edge_out_ch = edge_out_ch

        # Layers
        # ------
        # First layer is AlexNet Edge Extraction (with out bias)
        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=edge_out_ch,
            kernel_size=11,
            stride=4,
            padding=2,
            bias=False
        )
        # use pre-trained Alexnet weights
        alexnet_edge_detect_kernels = torchvision.models.alexnet(pretrained=True).features[0]
        self.conv1.weight.data.copy_(alexnet_edge_detect_kernels.weight.data)

        # Additional Batch normalization Layer
        self.bn1 = nn.BatchNorm2d(num_features=64)
        self.post = ClassifierHead(edge_out_ch)

    def forward(self, in_img):

        # Edge Extraction
        ff = self.conv1(in_img)
        ff = self.bn1(ff)
        ff = nn.functional.relu(ff)

        # Control - Contour Integration equivalent
        out_layer_arr = []

        # Post processing
        if self.get_iterative_predictions:
            out = self.post(ff), out_layer_arr
        else:
            out = self.post(ff)

        return out
