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
import torch.nn.init as init


# ---------------------------------------------------------------------------------------
#  Classifier Layers - Map to outputs
# ---------------------------------------------------------------------------------------
class DummyHead(nn.Module):
    """ Just passes the data through as is """
    def __init__(self, *argv):
        super(DummyHead, self).__init__()

    def forward(self, x):
        return x


class EdgeExtractClassifier(nn.Module):
    """
    Multiple conv layers that map to a single output channel without changing spatial
    dimensions. Up-sampling to original image size should be done before.

    Originally this was defined similar to the edge extract head of Doobnet. However,
    my own testing has found that performance drops but very little, if only
    2 layers (less than 1% IoU) if only two layers are used after contour integration
    """
    def __init__(self, n_in_channels):
        super(EdgeExtractClassifier, self).__init__()
        self.n_in_channels = n_in_channels

        self.conv_first = nn.Conv2d(
            in_channels=self.n_in_channels, out_channels=8, kernel_size=(3, 3), stride=(1, 1),
            padding=(1, 1), groups=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_features=8)

        # self.conv2 = nn.Conv2d(
        #     in_channels=8, out_channels=8, kernel_size=(3, 3), stride=(1, 1),
        #     padding=(1, 1), groups=1, bias=False)
        # self.bn2 = nn.BatchNorm2d(num_features=8)
        #
        # self.conv3 = nn.Conv2d(
        #     in_channels=8, out_channels=8, kernel_size=(3, 3), stride=(1, 1),
        #     padding=(1, 1), groups=1, bias=False)
        # self.bn3 = nn.BatchNorm2d(num_features=8)
        #
        # self.conv4 = nn.Conv2d(
        #     in_channels=8, out_channels=8, kernel_size=(3, 3), stride=(1, 1),
        #     padding=(1, 1), groups=1, bias=False)
        # self.bn4 = nn.BatchNorm2d(num_features=8)
        #
        # self.conv5 = nn.Conv2d(
        #     in_channels=8, out_channels=8, kernel_size=(3, 3), stride=(1, 1),
        #     padding=(1, 1), groups=1, bias=False)
        # self.bn5 = nn.BatchNorm2d(num_features=8)

        self.conv_final = nn.Conv2d(
            in_channels=8, out_channels=1, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)

    def forward(self, x):
        x = self.conv_first(x)
        x = self.bn1(x)
        x = nn.functional.relu(x)

        # x = self.conv2(x)
        # x = self.bn2(x)
        # x = nn.functional.relu(x)
        #
        # x = self.conv3(x)
        # x = self.bn3(x)
        # x = nn.functional.relu(x)
        #
        # x = self.conv4(x)
        # x = self.bn4(x)
        # x = nn.functional.relu(x)
        #
        # x = self.conv5(x)
        # x = self.bn5(x)
        # x = nn.functional.relu(x)

        x = self.conv_final(x)

        return x


class ClassifierHead(nn.Module):
    """
    This is designed to be used by a full model. It is designed to map to the
    output dimensions of the labels for the contour dataset. Currently the
    dimension matching is hard coded.
    """
    def __init__(self, num_channels):
        super(ClassifierHead, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels=num_channels,
            out_channels=num_channels // 4,
            kernel_size=(3, 3),
            stride=(2, 2),
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
            padding=(2, 2)
        )

    def forward(self, x):
        x = nn_functional.relu(self.bn1(self.conv1(x)))

        # Sigmoid is included with loss function (BCEWithLogitsLoss)
        # x = torch.sigmoid(self.conv2(x))
        x = self.conv2(x)

        return x


# ---------------------------------------------------------------------------------------
#  Contour Integration Layers
# ---------------------------------------------------------------------------------------

class CurrentSubtractInhibitLayer(nn.Module):
    def __init__(self, edge_out_ch=64, n_iters=5, lateral_e_size=7, lateral_i_size=7, a=None, b=None):
        """
        Contour Integration Layer - Current based with Subtractive Inhibition

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

        x_t = (1 - a)x_{t-1} +
              a ( J_{xx}f_x(x_{t-1}) - J_{xy}f_y(y_{t-1}) + e_bias + f_x(X_{t-1}) * W_e) + I_{ext} )

        y_t = (1 - b)y_{t-1} +
              b (J_{yx}f(x_{t-1}) + i_bias + f_y(Y_{t-1}) * W_i) )

        x = membrane potential of excitatory neuron
        y = membrane potential of the inhibitory neuron

        J_{xx} = connection strength of the recurrent self connection of the excitatory neuron.
        J_{xy} = connection strength from the inhibitory neuron to the excitatory neuron
        J_{yx} = connection strength from the excitatory neuron to the inhibitory neuron

        f_x(x) = Activation function of the excitatory neuron, Relu. It is the output of the same neuron
        f_y(y) = Activation function of the inhibitory neuron, Relu. It is the output of the same neuron

        X_{t} = Output of all excitatory neurons at time t
        Y_{y} = Output of all inhibitory neurons

        W_e = Connection strengths of nearby excitatory neurons to excitatory neurons.
        W_i = Connection strength of nearby excitatory neurons to inhibitory neurons.

        I_{ext} = External current = feed-forward input from edge extracting layer

        e_bias = Leakage, bias terms of excitatory neuron
        i_bias = Leakage, bias term of inhibitory neuron

        :param edge_out_ch:
        :param n_iters:
        :param lateral_e_size:
        :param lateral_i_size:
        """
        super(CurrentSubtractInhibitLayer, self).__init__()

        self.lateral_e_size = lateral_e_size
        self.lateral_i_size = lateral_i_size
        self.edge_out_ch = edge_out_ch
        self.n_iters = n_iters  # Number of recurrent steps

        # Parameters

        if a is not None:
            assert type(a) == float, 'a must be an float'
            assert 0 <= a <= 1.0, 'a must be between [0, 1]'

            self.a = nn.Parameter(torch.ones(edge_out_ch) * a)
            self.a.requires_grad = False
            self.fixed_a = a
        else:
            self.a = nn.Parameter(torch.rand(edge_out_ch))  # RV between [0, 1]

        if b is not None:
            assert type(b) == float, 'b must be an float'
            assert 0 <= b <= 1.0, 'b must be between [0, 1]'

            self.b = nn.Parameter(torch.ones(edge_out_ch) * b)
            self.b.requires_grad = False
            self.fixed_b = b
        else:
            self.b = nn.Parameter(torch.rand(edge_out_ch))  # RV between [0, 1]

        # Remove self excitation, a form is already included in the lateral connections
        # self.j_xx = nn.Parameter(torch.rand(edge_out_ch))
        # init.xavier_normal_(self.j_xx.view(1, edge_out_ch))

        self.j_xy = nn.Parameter(torch.rand(edge_out_ch))
        init.xavier_normal_(self.j_xy.view(1, edge_out_ch))

        self.j_yx = nn.Parameter(torch.rand(edge_out_ch))
        init.xavier_normal_(self.j_yx.view(1, edge_out_ch))

        self.e_bias = nn.Parameter(torch.ones(edge_out_ch)*0.01)
        self.i_bias = nn.Parameter(torch.ones(edge_out_ch)*0.01)

        # Components
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

    def forward(self, ff):
        """

        :param ff:  input from previous layer
        :return:
        """

        x = torch.zeros_like(ff)  # state of excitatory neurons
        f_x = torch.zeros_like(ff)  # Fire Rate (after nonlinear activation) of excitatory neurons
        y = torch.zeros_like(ff)  # state of inhibitory neurons
        f_y = torch.zeros_like(ff)  # Fire Rate (after nonlinear activation) of excitatory neurons

        # # Debug
        # idx = ff.argmax()  # This is the index in the flattened array

        for i in range(self.n_iters):
            # print("processing iteration {}".format(i))

            # # Debug
            # print("Start ff {:0.4f}, x {:0.4f}, f_x {:0.4f}, y {:0.4f}, f_y {:0.4f}, "
            #       "lat_e {:0.4f}, lat_i {:0.4f}".format(
            #     ff.flatten()[idx], x.flatten()[idx], f_x.flatten()[idx], y.flatten()[idx], f_y.flatten()[idx],
            #     nn.functional.relu(self.lateral_e(f_x)).flatten()[idx],
            #     nn.functional.relu(self.lateral_i(f_x)).flatten()[idx]))

            # crazy broadcasting. dim=1 tell torch that this dim needs to be broadcast
            gated_a = torch.sigmoid(self.a.view(1, self.edge_out_ch, 1, 1))
            gated_b = torch.sigmoid(self.b.view(1, self.edge_out_ch, 1, 1))

            sigmoid_j_xy = torch.sigmoid(self.j_xy.view(1, self.edge_out_ch, 1, 1))
            sigmoid_j_yx = torch.sigmoid(self.j_yx.view(1, self.edge_out_ch, 1, 1))

            x = (1 - gated_a) * x + \
                gated_a * (
                    # (self.j_xx.view(1, self.edge_out_ch, 1, 1) * f_x)
                    - (sigmoid_j_xy * f_y) +
                    ff +
                    self.e_bias.view(1, self.edge_out_ch, 1, 1) * torch.ones_like(ff) +
                    nn.functional.relu(self.lateral_e(f_x))
                )

            y = (1 - gated_b) * y + \
                gated_b * (
                    (sigmoid_j_yx * f_x) +
                    self.i_bias.view(1, self.edge_out_ch, 1, 1) * torch.ones_like(ff) +
                    nn.functional.relu(self.lateral_i(f_x))
                )

            f_x = nn.functional.relu(x)
            f_y = nn.functional.relu(y)

            # # Debug
            # print("Final iter {} x {:0.4f}, f_x {:0.4f}, y {:0.4f}, f_y {:0.4f}".format(
            #     i, x.flatten()[idx], f_x.flatten()[idx], y.flatten()[idx], f_y.flatten()[idx]))

        return f_x


class CurrentDivisiveInhibitLayer(nn.Module):
    def __init__(self, edge_out_ch=64, n_iters=5, lateral_e_size=7, lateral_i_size=7):
        """
        Current based model with Divisive Inhibition. See Piech-2013

        See CurrentSubtractInhibitLayer for more details.

        The main difference is how the inhibitory neuron interacts with the excitatory neurons.
        Here the impact is divisive rather than subtractive

        x_t = (1 - a) x_{t-1} +
              a (J_{xx}f_x(x_{t-1}) + e_bias + f_x(F_x(x_{t-1})* W_e + I_{ext}) /
              (1 + - J_{xy}f_y(y_{t-1}) )

        y_t = (1 - b)y_{t-1} +
              b (J_{yx}f(x_{t-1})+ i_bias + f_y(F_y(y_{t-1}) )


        :param edge_out_ch:
        :param n_iters:
        """

        super(CurrentDivisiveInhibitLayer, self).__init__()

        self.lateral_e_size = lateral_e_size
        self.lateral_i_size = lateral_i_size
        self.edge_out_ch = edge_out_ch
        self.n_iters = n_iters  # Number of recurrent steps

        # Parameters
        self.a = nn.Parameter(torch.rand(edge_out_ch))  # RV between [0, 1]
        self.b = nn.Parameter(torch.rand(edge_out_ch))  # RV between [0, 1]

        # # Remove self excitation, a form is already included in the lateral connections
        # self.j_xx = nn.Parameter(torch.rand(edge_out_ch))
        # init.xavier_normal_(self.j_xx.view(1, edge_out_ch))

        self.j_xy = nn.Parameter(torch.rand(edge_out_ch))
        init.xavier_normal_(self.j_xy.view(1, edge_out_ch))

        self.j_yx = nn.Parameter(torch.rand(edge_out_ch))
        init.xavier_normal_(self.j_yx.view(1, edge_out_ch))

        self.e_bias = nn.Parameter(torch.ones(edge_out_ch) * 0.01)
        self.i_bias = nn.Parameter(torch.ones(edge_out_ch) * 0.01)

        # Components
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

    def forward(self, ff):
        """

        :param ff:  input from previous layer
        :return:
        """

        x = torch.zeros_like(ff)  # state of excitatory neurons
        f_x = torch.zeros_like(ff)  # Fire Rate (after nonlinear activation) of excitatory neurons
        y = torch.zeros_like(ff)  # state of inhibitory neurons
        f_y = torch.zeros_like(ff)  # Fire Rate (after nonlinear activation) of excitatory neurons

        # # # Debug
        # idx = ff.argmax()  # This is the index in the flattened array

        for i in range(self.n_iters):
            # print("processing iteration {}".format(i))

            # # Debug
            # print("Start ff {:0.4f}, x {:0.4f}, f_x {:0.4f}, y {:0.4f}, f_y {:0.4f}, "
            #       "lat_e {:0.4f}, lat_i {:0.4f}".format(
            #         ff.flatten()[idx], x.flatten()[idx], f_x.flatten()[idx], y.flatten()[idx], f_y.flatten()[idx],
            #         nn.functional.relu(self.lateral_e(f_x)).flatten()[idx],
            #         nn.functional.relu(self.lateral_i(f_x)).flatten()[idx]))

            # crazy broadcasting. dim=1 tell torch that this dim needs to be broadcast
            gated_a = torch.sigmoid(self.a.view(1, self.edge_out_ch, 1, 1))
            gated_b = torch.sigmoid(self.b.view(1, self.edge_out_ch, 1, 1))

            sigmoid_j_xy = torch.sigmoid(self.j_xy.view(1, self.edge_out_ch, 1, 1))
            sigmoid_j_yx = torch.sigmoid(self.j_yx.view(1, self.edge_out_ch, 1, 1))

            x = (1 - gated_a) * x + \
                gated_a * (
                    ff
                    + (self.e_bias.view(1, self.edge_out_ch, 1, 1) * torch.ones_like(ff))
                    + nn.functional.relu(self.lateral_e(f_x))
                    # + (self.j_xx.view(1, self.edge_out_ch, 1, 1) * f_x)
                ) / (1 + (sigmoid_j_xy * f_y))

            y = (1 - gated_b) * y \
                + gated_b * (
                    (sigmoid_j_yx * f_x)
                    + self.i_bias.view(1, self.edge_out_ch, 1, 1) * torch.ones_like(ff)
                    + nn.functional.relu(self.lateral_i(f_x))
                )

            f_x = nn.functional.relu(x)
            f_y = nn.functional.relu(y)

            # # Debug
            # print("Final iter {} x {:0.4f}, f_x {:0.4f}, y {:0.4f}, f_y {:0.4f}".format(
            #     i, x.flatten()[idx], f_x.flatten()[idx], y.flatten()[idx], f_y.flatten()[idx]))

        return f_x


# ---------------------------------------------------------------------------------------
# Full Models
# ---------------------------------------------------------------------------------------

class ContourIntegrationAlexnet(nn.Module):
    """
    Minimal Model with contour integration layer for the contour dataset
    First (Edge extracting) layer of Alexnet
    """
    def __init__(self, contour_integration_layer, pre_trained_edge_extract=True, classifier=ClassifierHead):

        super(ContourIntegrationAlexnet, self).__init__()

        self.pre_trained_edge_extract = pre_trained_edge_extract

        # First Convolutional Layer of Alexnet
        # self.edge_extract = torchvision.models.alexnet(pretrained= self.pre_trained_edge_extract).features[0]

        self.edge_extract = nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2, bias=False)
        alexnet_kernel = torchvision.models.alexnet(pretrained=self.pre_trained_edge_extract).features[0]
        if self.pre_trained_edge_extract:
            self.edge_extract.weight.requires_grad = False
            self.edge_extract.weight.data = alexnet_kernel.weight.data
        else:
            init.xavier_normal_(self.edge_extract.weight)

        self.num_edge_extract_chan = self.edge_extract.weight.shape[0]

        self.bn1 = nn.BatchNorm2d(num_features=self.num_edge_extract_chan)

        self.contour_integration_layer = contour_integration_layer

        self.classifier = classifier(self.num_edge_extract_chan)

    def forward(self, in_img):

        # Edge Extraction
        x = self.edge_extract(in_img)
        x = self.bn1(x)
        x = nn.functional.relu(x)

        x = self.contour_integration_layer(x)

        x = self.classifier(x)

        return x


class ContourIntegrationResnet50(nn.Module):
    """
    Minimal Model with contour integration layer for the contour dataset
    This one includes the first layers of a Resnet50
    """

    def __init__(self, contour_integration_layer, pre_trained_edge_extract=True, classifier=ClassifierHead):
        super(ContourIntegrationResnet50, self).__init__()

        self.pre_trained_edge_extract = pre_trained_edge_extract

        self.edge_extract = torchvision.models.resnet50(pretrained=self.pre_trained_edge_extract).conv1
        if self.pre_trained_edge_extract:
            self.edge_extract.weight.requires_grad = False
        else:
            init.xavier_normal_(self.edge_extract.weight)

        self.num_edge_extract_chan = self.edge_extract.weight.shape[0]
        self.bn1 = nn.BatchNorm2d(num_features=self.num_edge_extract_chan)

        self.max_pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.contour_integration_layer = contour_integration_layer

        self.classifier = classifier(self.num_edge_extract_chan)

    def forward(self, in_img):

        # Edge Extraction
        x = self.edge_extract(in_img)
        x = self.bn1(x)
        x = nn.functional.relu(x)
        x = self.max_pool1(x)
        # The above is directly from Resnet50. Attaching the contour integration layer here (as opposed to
        # after the conv layer) since this has the same dimensions as alexnet edge extract. Allows to use
        # the same classification head.

        x = self.contour_integration_layer(x)

        x = self.classifier(x)

        return x


def get_embedded_resnet50_model(saved_contour_integration_model=None, pretrained=True):
    """
    Returns a Full Resnet50 Model with a contour integration layer embedded in it

    :param saved_contour_integration_model:
    :param pretrained:
    :return:
    """

    model = torchvision.models.resnet50(pretrained=pretrained)

    cont_int_layer = CurrentSubtractInhibitLayer(
        lateral_e_size=15, lateral_i_size=15, n_iters=5,)

    # We always want the first layer (edge extraction) to be pre trained
    cont_int_model = ContourIntegrationResnet50(
        contour_integration_layer=cont_int_layer,
        classifier=DummyHead,
        pre_trained_edge_extract=True
    )

    if saved_contour_integration_model is not None:
        cont_int_model.load_state_dict(torch.load(saved_contour_integration_model), strict=False)
        # strict = False do not care about loading classifier weights

    # The Resnet50 Contour integration Model attaches after the first max pooling layer
    # of resent. Replace one layer of Resnet50 and force all others to be dummy Heads (pass through as is)
    model.conv1 = cont_int_model
    model.bn1 = DummyHead()
    model.relu = DummyHead()
    model.maxpool = DummyHead()

    return model


class EdgeDetectionResnet50(nn.Module):
    """
       Model for the edge detection Dataset
    """
    def __init__(self, contour_integration_layer, pre_trained_edge_extract=True):
        super(EdgeDetectionResnet50, self).__init__()

        self.pre_trained_edge_extract = pre_trained_edge_extract

        self.edge_extract = torchvision.models.resnet50(pretrained=self.pre_trained_edge_extract).conv1
        if self.pre_trained_edge_extract:
            self.edge_extract.weight.requires_grad = False
        else:
            init.xavier_normal_(self.edge_extract.weight)

        self.num_edge_extract_chan = self.edge_extract.weight.shape[0]
        self.bn1 = nn.BatchNorm2d(num_features=self.num_edge_extract_chan)

        self.contour_integration_layer = contour_integration_layer

        self.classifier = EdgeExtractClassifier(n_in_channels=self.num_edge_extract_chan)

    def forward(self, in_img):

        img_size = in_img.shape[2:]

        # Edge Extraction
        x = self.edge_extract(in_img)
        x = self.bn1(x)
        x = nn.functional.relu(x)

        x = self.contour_integration_layer(x)

        # Up sample to the input image size
        x = nn.functional.interpolate(x, size=img_size, mode='bilinear')

        x = self.classifier(x)

        return x
