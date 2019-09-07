# ------------------------------------------------------------------------------------
# Contour Integration Models based on
#
#    Piech, Li, Reeke & Gilbert - 2013 -
#    Network Model of Top-down influences on local gain and contextual interactions.
#
# Three models are defined:
#       [1] Current based with subtractive inhibition
#       [2] Current based with divisive inhibition
#       [3] Conductance based with subtractive inhibition
#       TODO: Add Conductance based model
# ------------------------------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as nn_functional
import torchvision


class DummyHead(nn.Module):
    """ Just passes the data through as is """
    def __init__(self):
        super(DummyHead, self).__init__()

    def forward(self, x):
        return x


class ClassifierHead(nn.Module):
    def __init__(self, num_channels):
        super(ClassifierHead, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels=num_channels,
            out_channels=num_channels // 4,
            kernel_size=(3, 3),
            stride=(4, 4),
            groups=1,
            bias=False,
            padding=(2, 2)
        )
        self.bn1 = nn.BatchNorm2d(num_channels // 4)

        self.conv2 = nn.Conv2d(
            in_channels=num_channels // 4,
            out_channels=1,
            kernel_size=(1, 1),
            stride=(2, 2),
            groups=1,
            bias=False,
        )

    def forward(self, x):
        x = nn_functional.relu(self.bn1(self.conv1(x)))
        x = torch.sigmoid(self.conv2(x))
        return x


class ClassifierHeadOld(nn.Module):
    def __init__(self, num_channels):
        super(ClassifierHeadOld, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels=num_channels,
            out_channels=num_channels // 2,
            kernel_size=(3, 3),
            stride=(2, 2),
            groups=1,
            bias=False,
            padding=(2, 2)
        )

        self.bn1 = nn.BatchNorm2d(num_channels // 2)

        self.conv2 = nn.Conv2d(
            in_channels=num_channels // 2,
            out_channels=num_channels // 4,
            kernel_size=(3, 3),
            stride=(2, 2),
            groups=1,
            padding=(1, 1),
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(num_channels // 4)

        self.conv3 = nn.Conv2d(
            in_channels=num_channels // 4,
            out_channels=1,
            kernel_size=(1, 1),
            stride=(2, 2),
            groups=1,
            bias=False,
        )

    def forward(self, x):
        x = nn_functional.relu(self.bn1(self.conv1(x)))
        x = nn_functional.relu(self.bn2(self.conv2(x)))
        x = torch.sigmoid(self.conv3(x))

        return x


class CurrentSubtractiveInhibition(nn.Module):
    def __init__(self, edge_out_ch=64, n_iters=5, use_class_head=True, lateral_e_size=7, lateral_i_size=7):
        """
        Current based model with Subtractive Inhibition

        The basic unit of the contour integration layer is a neuron pair, a set of excitatory (x) and inhibitory (y)
        neurons. A common neuron pair is defined for each feature channel (edge_out_ch). This are then applied at all
        spatial locations.

        In current-based models, influences from different components are constant. They are not dependent on
        membrane potential. For Example the influence from the inhibitory neuron is -J_{xy}. While for a conductance-
        based model, the influence is variable as it depends on the membrane potential, -g_{xy}.(V_equ - x). In
        conductance based models, the effective strength of the connection weakens near the equilibrium potential
        Here, J represent current, g is conductance, x = membrane potential and V_equ is the equilibrium potential.

        Subtractive Inhibition means the inhibitory neurons influence is subtracted from the membrane potential of the
        excitatory neuron.

        x_t = (1 - a) x_{t-1} +
              a (J_{xx}f_x(x_{t-1}) - J_{xy}f_y(y_{t-1}) + e_bias + f_x(F_x(x_{t-1})* W_e + I_{ext} )

        y_t = (1 - b)y_{t-1} +
              b (J_{yx}f(x_{t-1})+ i_bias + f_y(F_y(y_{t-1}) )

        x = membrane potential of excitatory neuron
        y = membrane potential of the inhibitory neuron

        J_{xx} = connection strength of the recurrent self connection of the excitatory neuron.
        J_{xy} = connection strength from the inhibitory neuron to the excitatory neuron
        J_{yx} = connection strength from the excitatory neuron to the inhibitory neuron

        f_x(x) = Activation function of the excitatory neuron, Relu. It is the output of the same neuron
        f_y(y) = Activation function of the inhibitory neuron, Relu. It is the output of the same neuron

        F_x(x) = Output of all excitatory neurons.
        F_y(y) = Output of all inhibitory neurons

        W_e = Connection strengths of nearby excitatory neurons to excitatory neurons.
        W_i = Connection strength of nearby excitatory neurons to inhibitory neurons.

        I_{ext} = External current = feed-forward input from edge extracting layer

        e_bias = Leakage, bias terms of excitatory neuron
        i_bias = Leakage, bias term of inhibitory neuron

        :param edge_out_ch: Number of output channels
        :param n_iters: number of recurrent iteration steps.

        """

        super(CurrentSubtractiveInhibition, self).__init__()

        # If set, will additionally return the predictions of the contour integration layers activation after each
        # iteration (these are passed though the classification head)
        self.get_iterative_predictions = False
        # If True, use classification head to get predictions or if False, output will feed into another layer
        # the dimensions of the edge extracting layer will be preserved
        self.use_class_head = use_class_head
        self.lateral_e_size = lateral_e_size
        self.lateral_i_size = lateral_i_size

        # Parameters
        self.n_iters = n_iters  # Number of recurrent steps
        self.a = 0.5  # Weighting factor for combining excitatory recurrence and feed-forward. Should be [Nx1]
        self.b = 0.5  # Weighting factor for combining inhibitory recurrence and feed-forward  Should be [Nx1]
        # TODO: a and b should be learnt not constant

        self.edge_out_ch = edge_out_ch
        self.a = nn.Parameter(torch.rand(edge_out_ch))
        self.b = nn.Parameter(torch.rand(edge_out_ch))

        self.j_xx = nn.Parameter(torch.rand(edge_out_ch))
        self.j_xy = nn.Parameter(torch.rand(edge_out_ch))
        self.j_yx = nn.Parameter(torch.rand(edge_out_ch))

        self.e_bias = nn.Parameter(torch.rand(edge_out_ch))
        self.i_bias = nn.Parameter(torch.rand(edge_out_ch))

        # Layers
        # -------------------------------------------------------------------------------

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
        self.conv1.weight.requires_grad = False

        # Additional batch normalization Layer
        self.bn1 = nn.BatchNorm2d(num_features=self.edge_out_ch)

        # Contour Integration Layer
        # TODO: What is the spatial extent of the kernel for one iteration
        self.lateral_e = nn.Conv2d(
            in_channels=edge_out_ch,
            out_channels=self.edge_out_ch,
            kernel_size=self.lateral_e_size,
            stride=1,
            padding=(self.lateral_e_size - 1) // 2,  # Keep the input dimensions
            bias=False
        )
        self.lateral_i = nn.Conv2d(
            in_channels=edge_out_ch,
            out_channels=self.edge_out_ch,
            kernel_size=self.lateral_i_size,
            stride=1,
            padding=(self.lateral_i_size - 1) // 2,  # Keep the input dimensions
            bias=False
        )

        # Classification head get decision (whether part of a contour or not) for each full tile in the image.
        if use_class_head:
            self.post = ClassifierHead(self.edge_out_ch)
        else:
            self.post = DummyHead()

    def forward(self, in_img):

        # Edge Extraction
        ff = self.conv1(in_img)
        ff = self.bn1(ff)
        ff = nn.functional.relu(ff)

        # -------------------------------------------------------------------------------
        # Contour Integration
        # -------------------------------------------------------------------------------
        x = torch.zeros_like(ff)    # state of excitatory neurons
        f_x = torch.zeros_like(ff)  # Fire Rate (after nonlinear activation) of excitatory neurons
        y = torch.zeros_like(ff)    # state of inhibitory neurons
        f_y = torch.zeros_like(ff)  # Fire Rate (after nonlinear activation) of excitatory neurons

        iterative_out_arr = []

        for i in range(self.n_iters):

            # print("processing iteration {}".format(i))

            # crazy broadcasting. dim=1 tell torch that this dim needs to be broadcast
            x = (1 - self.a.view(1, self.edge_out_ch, 1, 1)) * x + \
                self.a.view(1, self.edge_out_ch, 1, 1) * (
                    (self.j_xx.view(1, self.edge_out_ch, 1, 1) * f_x) -
                    (self.j_xy.view(1, self.edge_out_ch, 1, 1) * f_y) +
                    ff +
                    self.e_bias.view(1, self.edge_out_ch, 1, 1) * torch.ones_like(ff) +
                    nn.functional.relu(self.lateral_e(f_x))
                )
            # TODO: first f_x should be one dimensional for each channel, second one should include neighbors

            f_x = nn.functional.relu(x)

            y = (1 - self.b.view(1, self.edge_out_ch, 1, 1) * y) + \
                self.b.view(1, self.edge_out_ch, 1, 1) * (
                    (self.j_yx.view(1, self.edge_out_ch, 1, 1) * f_x) +
                    self.i_bias.view(1, self.edge_out_ch, 1, 1) * torch.ones_like(ff) +
                    nn.functional.relu(self.lateral_i(f_x))
                )

            f_y = nn.functional.relu(y)

            if self.get_iterative_predictions:
                iterative_out_arr.append(self.post(f_x))

        # Post processing
        if self.get_iterative_predictions:
            out = self.post(f_x), iterative_out_arr
        else:
            out = self.post(f_x)

        return out


class CurrentDivisiveInhibition(nn.Module):
    def __init__(self, edge_out_ch=64, n_iters=5, use_class_head=True, lateral_e_size=7, lateral_i_size=7):
        """
        Current based model with Divisive Inhibition

        See CurrentSubtractiveInhibition model for more details.

        The main difference is how the inhibitory neuron interacts with the excitatory neurons.
        Here the impact is divisive rather than subtractive


        x_t = (1 - a) x_{t-1} +
              a (J_{xx}f_x(x_{t-1}) + e_bias + f_x(F_x(x_{t-1})* W_e + I_{ext}) / (1+ - J_{xy}f_y(y_{t-1}) )

        y_t = (1 - b)y_{t-1} +
              b (J_{yx}f(x_{t-1})+ i_bias + f_y(F_y(y_{t-1}) )


        :param edge_out_ch:
        :param n_iters:
        """

        super(CurrentDivisiveInhibition, self).__init__()

        # If set, will additionally return the predictions of the contour integration layers activation after each
        # iteration (these are passed though the classification head)
        self.get_iterative_predictions = False
        # If True, use classification head to get predictions or if False, output will feed into another layer
        # the dimensions of the edge extracting layer will be preserved
        self.use_class_head = use_class_head
        self.lateral_e_size = lateral_e_size
        self.lateral_i_size = lateral_i_size

        # Parameters
        self.n_iters = n_iters  # Number of recurrent steps
        self.a = 0.5  # Weighting factor for combining excitatory recurrence and feed-forward. Should be [Nx1]
        self.b = 0.5  # Weighting factor for combining inhibitory recurrence and feed-forward  Should be [Nx1]
        # N = number of feature channels
        # TODO: 1. find proper setting, 2. can these be learned?

        self.edge_out_ch = edge_out_ch
        self.a = nn.Parameter(torch.rand(edge_out_ch))
        self.b = nn.Parameter(torch.rand(edge_out_ch))

        self.j_xx = nn.Parameter(torch.rand(edge_out_ch))
        self.j_xy = nn.Parameter(torch.rand(edge_out_ch))
        self.j_yx = nn.Parameter(torch.rand(edge_out_ch))

        self.e_bias = nn.Parameter(torch.rand(edge_out_ch))
        self.i_bias = nn.Parameter(torch.rand(edge_out_ch))

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
        self.bn1 = nn.BatchNorm2d(num_features=self.edge_out_ch)

        # Recurrent lateral connection parts
        # TODO: What is the spatial extent of the kernel for one iteration
        # TODO: Adjust padding to match dimensions
        self.lateral_e = nn.Conv2d(
            in_channels=edge_out_ch,
            out_channels=self.edge_out_ch,
            kernel_size=self.lateral_e_size,
            stride=1,
            padding=(self.lateral_e_size - 1) // 2,  # Keep the input dimensions
            bias=False
        )
        self.lateral_i = nn.Conv2d(
            in_channels=edge_out_ch,
            out_channels=self.edge_out_ch,
            kernel_size=self.lateral_i_size,
            stride=1,
            padding=(self.lateral_i_size - 1) // 2,  # Keep the input dimensions
            bias=False
        )

        # Classification head get decision (whether part of a contour or not) for each full tile in the image.
        if use_class_head:
            self.post = ClassifierHead(edge_out_ch)
        else:
            self.post = DummyHead()

    def forward(self, in_img):

        # Edge Extraction
        ff = self.conv1(in_img)
        ff = self.bn1(ff)
        ff = nn.functional.relu(ff)

        # -------------------------------------------------------------------------------
        # Contour Integration
        # -------------------------------------------------------------------------------
        x = torch.zeros_like(ff)    # state of excitatory neurons
        f_x = torch.zeros_like(ff)  # Fire Rate (after nonlinear activation) of excitatory neurons
        y = torch.zeros_like(ff)    # state of inhibitory neurons
        f_y = torch.zeros_like(ff)  # Fire Rate (after nonlinear activation) of excitatory neurons

        iterative_out_arr = []

        for i in range(self.n_iters):

            # print("processing iteration {}".format(i))

            # crazy broadcasting. dim=1 tell torch that this dim needs to be broadcast
            x = (1 - self.a.view(1, self.edge_out_ch, 1, 1)) * x + \
                self.a.view(1, self.edge_out_ch, 1, 1) * (
                    (self.j_xx.view(1, self.edge_out_ch, 1, 1) * f_x)
                    + ff
                    # - (self.j_xy.view(1, self.edge_out_ch, 1, 1) * f_y)
                    + self.e_bias.view(1, self.edge_out_ch, 1, 1) * torch.ones_like(ff)
                    + nn.functional.relu(self.lateral_e(f_x))
                ) / (1 + (self.j_xy.view(1, self.edge_out_ch, 1, 1) * f_y))
            # TODO: first f_x should be one dimensional for each channel, second one should include neighbors

            f_x = nn.functional.relu(x)

            y = (1 - self.b.view(1, self.edge_out_ch, 1, 1) * y) + \
                self.b.view(1, self.edge_out_ch, 1, 1) * (
                    (self.j_yx.view(1, self.edge_out_ch, 1, 1) * f_x) +
                    self.i_bias.view(1, self.edge_out_ch, 1, 1) * torch.ones_like(ff) +
                    nn.functional.relu(self.lateral_i(f_x))
                )

            f_y = nn.functional.relu(y)

            if self.get_iterative_predictions:
                iterative_out_arr.append(self.post(f_x))

        # Post processing
        if self.get_iterative_predictions:
            out = self.post(f_x), iterative_out_arr
        else:
            out = self.post(f_x)

        return out
