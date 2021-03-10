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


class BinaryClassifier(nn.Module):
    def __init__(self, n_in_channels):
        """
        Final output is a single value.
        Given input dimension [CH, R, C], 3 conv layers are applied.

        First one reduces CH/2, R/2, C/2
        Second one reduces CH/4, R/4, C/4
        Third one reduces  1, R/4, C/4,
        Global Avg Pool:  [1, 1, 1] averages over all R/4, C/4 to give 1 output.

        Sigmoid is not applied as it is included with the loss

        :param n_in_channels:
        """
        super(BinaryClassifier, self).__init__()
        self.n_in_channels = n_in_channels

        self.conv_first = nn.Conv2d(
            in_channels=self.n_in_channels, out_channels=self.n_in_channels // 2, kernel_size=(3, 3), stride=(2, 2),
            padding=(1, 1), groups=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_features=self.n_in_channels // 2)

        self.conv_second = nn.Conv2d(
            in_channels=self.n_in_channels // 2, out_channels=self.n_in_channels // 4, kernel_size=(3, 3), stride=(2, 2),
            padding=(1, 1), groups=1, bias=False)
        self.bn2 = nn.BatchNorm2d(num_features=self.n_in_channels // 4)

        self.conv_final = nn.Conv2d(
            in_channels=self.n_in_channels // 4, out_channels=1, kernel_size=(1, 1), bias=True)

        self.avg_pool = nn.AvgPool2d(n_in_channels // 4)

    def forward(self, x):
        x = self.conv_first(x)
        x = self.bn1(x)
        x = nn.functional.relu(x)

        x = self.conv_second(x)
        x = self.bn2(x)
        x = nn.functional.relu(x)

        x = self.conv_final(x)
        x = self.avg_pool(x)

        # Done per dimension to avoid the problem of zero dimension output
        # if batch size = 1 (see torch.squeeze docs)
        x = torch.squeeze(x, 3)
        x = torch.squeeze(x, 2)
        x = torch.squeeze(x, 1)

        return x


# class BinaryClassifier(nn.Module):
#     def __init__(self, n_in_channels, final_conv_dim=43):
#         super(BinaryClassifier, self).__init__()
#         self.n_in_channels = n_in_channels
#
#         self.conv_first = nn.Conv2d(
#             in_channels=self.n_in_channels, out_channels=8, kernel_size=(3, 3), stride=(3, 3),
#             padding=(1, 1), groups=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(num_features=8)
#
#         self.conv_final = nn.Conv2d(
#             in_channels=8, out_channels=1, kernel_size=(1, 1), bias=True)
#
#         self.avg_pool = nn.AvgPool2d(final_conv_dim)  # 43=C=R
#
#     def forward(self, x):
#         x = self.conv_first(x)
#         x = self.bn1(x)
#         x = nn.functional.relu(x)
#
#         x = self.conv_final(x)
#         x = self.avg_pool(x)
#
#         # Done per dimension to avoid the problem of zero dimension output
#         # if batch size = 1 (see torch.squeeze docs)
#         x = torch.squeeze(x, 3)
#         x = torch.squeeze(x, 2)
#         x = torch.squeeze(x, 1)
#
#         return x


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
def threshold_relu(input1, thres=0.1, below_value=0, above_thres_multiplier=1):
    """
    Shift relu with variable slope.

    :param input1: input to the model
    :param thres: Threshold at which slope becomes non-zero
    :param below_value: output value if input is below thresh
    :param above_thres_multiplier: A slop multiplier

    :return:
    """
    return torch.nn.functional.threshold(
        above_thres_multiplier*input1, threshold=thres, value=below_value, inplace=True)


class RecurrentBatchNorm(nn.Module):
    def __init__(self, num_features, n_iters, eps=1e-5, momentum=0.1, affine=True):
        """
        Recurrent Batch Normalization base on [Coolijmans et. al. - 2017 - Recurrent
        Batch Normalization]

        Code Ref:
        https://github.com/jihunchoi/recurrent-batch-normalization-pytorch/blob/master/bnlstm.py

        Tracks Mean and Variance separately per time steps
        Gamma and Beta Parameters are same per Iteration

        :param num_features:
        :param n_iters:
        :param eps:
        :param momentum:
        :param affine:
        """
        super(RecurrentBatchNorm, self).__init__()

        self.num_features = num_features
        self.n_iters = n_iters
        self.affine = affine
        self.eps = eps
        self.momentum = momentum

        if self.affine:
            self.weight = nn.Parameter(torch.FloatTensor(num_features))
            self.bias = nn.Parameter(torch.FloatTensor(num_features))
        else:
            # Still a parameter but it is None.
            # Can assign values to it, so equations will work. But will not be maintained
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

        for i in range(n_iters):
            self.register_buffer(
                'running_mean_{}'.format(i), torch.zeros(num_features))
            self.register_buffer(
                'running_var_{}'.format(i), torch.ones(num_features))
        self.reset_parameters()

    def reset_parameters(self):
        for i in range(self.n_iters):
            running_mean_i = getattr(self, 'running_mean_{}'.format(i))
            running_var_i = getattr(self, 'running_var_{}'.format(i))
            running_mean_i.zero_()
            running_var_i.fill_(1)

        if self.affine:
            self.weight.data.uniform_()
            self.bias.data.zero_()

    def _check_input_dim(self, input_):
        if input_.size(1) != self.running_mean_0.nelement():
            raise ValueError(
                'got {}-feature tensor, expected {}'.format(input_.size(1), self.num_features))

    def forward(self, input_, time):
        self._check_input_dim(input_)

        if time >= self.n_iters:
            time = self.n_iters - 1

        running_mean = getattr(self, 'running_mean_{}'.format(time))
        running_var = getattr(self, 'running_var_{}'.format(time))

        return torch.nn.functional.batch_norm(
            input=input_, running_mean=running_mean, running_var=running_var,
            weight=self.weight, bias=self.bias, training=self.training,
            momentum=self.momentum, eps=self.eps)

    def __repr__(self):
        return ('{name}({num_features}, eps={eps}, momentum={momentum},'
                ' n_iters={n_iters}, affine={affine})'
                .format(name=self.__class__.__name__, **self.__dict__))


# ---------------------------------------------------------------------------------------
#  Contour Integration Layers
# ---------------------------------------------------------------------------------------
class CurrentSubtractInhibitLayer(nn.Module):
    def __init__(self, edge_out_ch=64, n_iters=5, lateral_e_size=7, lateral_i_size=7, a=None, b=None,
                 j_xy=None, j_yx=None, use_recurrent_batch_norm=False, store_recurrent_acts=False):
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
        (Note see param notes to details on how they are actually used. A sigmoid is applied on their
        value before applying)

        f_x(x) = Activation function of the excitatory neuron, Relu. It is the output of the same neuron
        f_y(y) = Activation function of the inhibitory neuron, Relu. It is the output of the same neuron

        X_{t} = Output of all excitatory neurons at time t
        Y_{y} = Output of all inhibitory neurons

        W_e = Connection strengths of nearby excitatory neurons to excitatory neurons.
        W_i = Connection strength of nearby excitatory neurons to inhibitory neurons.

        I_{ext} = External current = feed-forward input from edge extracting layer

        e_bias = Leakage, bias terms of excitatory neuron
        i_bias = Leakage, bias term of inhibitory neuron

        :param edge_out_ch: Number of input and output channels.
        :param n_iters: Number of recurrent steps
        :param lateral_e_size:
        :param lateral_i_size:
        :param a: Excitatory mixing with previous activation. Note that a sigmoid is applied on
                  a before it interacts with the rest of the model. sigma(a) = 1/tau_a in the original equation.
                  If not specified [Default]. Random values between 0 and 1 are chosen.
        :param b: Inhibitory mixing with previous activation. Note that a sigmoid is applied on
                  b before it interacts with the rest of the model. sigma(b) = 1/tau_b in the original equation.
                  If not specified [Default]. Random values between 0 and 1 are chosen.

        :param j_xy: Connection Strength from inhibitory to Excitatory
        :param j_yx: Connection Strength from excitatory to Inhibitory Node
        :param store_recurrent_acts: If true store, recurrent X and Y activations per iteration.
                  Only for debugging
        """
        super(CurrentSubtractInhibitLayer, self).__init__()

        self.lateral_e_size = lateral_e_size
        self.lateral_i_size = lateral_i_size
        self.edge_out_ch = edge_out_ch
        self.n_iters = n_iters
        self.use_recurrent_batch_norm = use_recurrent_batch_norm

        # Only for Debugging
        self.store_recurrent_acts = store_recurrent_acts
        if self.store_recurrent_acts:
            self.x_per_iteration = []
            self.y_per_iteration = []

        # Parameters
        if a is not None:
            self.a = nn.Parameter(torch.ones(edge_out_ch) * a, requires_grad=False)
        else:
            self.a = nn.Parameter(torch.rand(edge_out_ch), requires_grad=True)  # RV between [0, 1]

        if b is not None:
            self.b = nn.Parameter(torch.ones(edge_out_ch) * b, requires_grad=False)
        else:
            self.b = nn.Parameter(torch.rand(edge_out_ch), requires_grad=True)  # RV between [0, 1]

        # Remove self excitation, a form is already included in the lateral connections
        # self.j_xx = nn.Parameter(torch.rand(edge_out_ch))
        # init.xavier_normal_(self.j_xx.view(1, edge_out_ch))

        if j_xy is not None:
            self.j_xy = nn.Parameter(torch.ones(edge_out_ch) * j_xy, requires_grad=False)
        else:
            # self.j_xy = nn.Parameter(torch.rand(edge_out_ch), requires_grad=True)
            # init.xavier_normal_(self.j_xy.view(1, edge_out_ch))

            # Empirically found this initialization, results in good IoU and gain curves
            self.j_xy = nn.Parameter(torch.ones(edge_out_ch) * -2.2, requires_grad=True)

        if j_yx is not None:
            self.j_yx = nn.Parameter(torch.ones(edge_out_ch) * j_yx, requires_grad=False)
        else:
            # self.j_yx = nn.Parameter(torch.rand(edge_out_ch), requires_grad=True)
            # init.xavier_normal_(self.j_yx.view(1, edge_out_ch))

            # Empirically found this initialization, results in good IoU and gain curves
            self.j_yx = nn.Parameter(torch.ones(edge_out_ch)*-2.2, requires_grad=True)

        self.e_bias = nn.Parameter(torch.ones(edge_out_ch)*0.01, requires_grad=True)
        self.i_bias = nn.Parameter(torch.ones(edge_out_ch)*0.01, requires_grad=True)

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

        if use_recurrent_batch_norm:
            self.recurrent_e_BN = nn.ModuleList([])
            self.recurrent_i_BN = nn.ModuleList([])
            for i in range(self.n_iters):
                self.recurrent_e_BN.append(torch.nn.BatchNorm2d(num_features=self.edge_out_ch))
                self.recurrent_i_BN.append(torch.nn.BatchNorm2d(num_features=self.edge_out_ch))

            # # Recurrent batch norm as defined in [Coolijmans et. al. - 2017]
            # # Single set of alpha\beta but different tracked running mean and var per iteration
            # self.recurrent_e_BN = RecurrentBatchNorm(self.edge_out_ch, self.n_iters)
            # self.recurrent_i_BN = RecurrentBatchNorm(self.edge_out_ch, self.n_iters)
            #
            # # Initialization of Batch normalization layer as defined in  [Coolijmans et. al. - 2017]
            # # No bias and alpha init to 0.1
            # self.recurrent_e_BN.weight.data.fill_(0.1)
            # self.recurrent_e_BN.bias.data.fill_(0)
            # self.recurrent_e_BN.bias.requires_grad = False
            #
            # self.recurrent_i_BN.weight.data.fill_(0.1)
            # self.recurrent_i_BN.bias.data.fill_(0)
            # self.recurrent_i_BN.bias.requires_grad = False

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

            if self.use_recurrent_batch_norm:
                f_x = nn.functional.relu(self.recurrent_e_BN[i](x))
                f_y = nn.functional.relu(self.recurrent_i_BN[i](y))
                # f_x = nn.functional.relu(self.recurrent_e_BN(x, i))
                # f_y = nn.functional.relu(self.recurrent_i_BN(y, i))
                
            else:
                f_x = nn.functional.relu(x)
                f_y = nn.functional.relu(y)
                # f_x = threshold_relu(x, thres=0.1, below_value=0, above_thres_multiplier=1)
                # f_y = threshold_relu(y, thres=0.1, below_value=0, above_thres_multiplier=2)

            if self.store_recurrent_acts:
                self.x_per_iteration.append(x)
                self.y_per_iteration.append(y)

            # # Debug
            # print("Final iter {} x {:0.4f}, f_x {:0.4f}, y {:0.4f}, f_y {:0.4f}".format(
            #     i, x.flatten()[idx], f_x.flatten()[idx], y.flatten()[idx], f_y.flatten()[idx]))

        return f_x


class CurrentDivisiveInhibitLayer(nn.Module):
    def __init__(self, edge_out_ch=64, n_iters=5, lateral_e_size=7, lateral_i_size=7, a=None, b=None,
                 j_xy=None, j_yx=None, use_recurrent_batch_norm=False, store_recurrent_acts=False):
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

        x = membrane potential of excitatory neuron
        y = membrane potential of the inhibitory neuron

        J_{xx} = connection strength of the recurrent self connection of the excitatory neuron.
        J_{xy} = connection strength from the inhibitory neuron to the excitatory neuron
        J_{yx} = connection strength from the excitatory neuron to the inhibitory neuron
        (Note see param notes to details on how they are actually used. A sigmoid is applied on their
        value before applying)

        f_x(x) = Activation function of the excitatory neuron, Relu. It is the output of the same neuron
        f_y(y) = Activation function of the inhibitory neuron, Relu. It is the output of the same neuron

        X_{t} = Output of all excitatory neurons at time t
        Y_{y} = Output of all inhibitory neurons

        W_e = Connection strengths of nearby excitatory neurons to excitatory neurons.
        W_i = Connection strength of nearby excitatory neurons to inhibitory neurons.

        I_{ext} = External current = feed-forward input from edge extracting layer

        e_bias = Leakage, bias terms of excitatory neuron
        i_bias = Leakage, bias term of inhibitory neuron

        :param edge_out_ch: Number of input and output channels.
        :param n_iters: Number of recurrent steps
        :param lateral_e_size:
        :param lateral_i_size:
        :param a: Excitatory mixing with previous activation. Note that a sigmoid is applied on
                  a before it interacts with the rest of the model. sigma(a) = 1/tau_a in the original equation.
                  If not specified [Default]. Random values between 0 and 1 are chosen. Note a sigmoid
                  is used before applying.
        :param b: Inhibitory mixing with previous activation. Note that a sigmoid is applied on
                  b before it interacts with the rest of the model. sigma(b) = 1/tau_b in the original equation.
                  If not specified [Default]. Random values between 0 and 1 are chosen. Note a sigmoid is used before applying.

        :param j_xy: Connection Strength from inhibitory to Excitatory
        :param j_yx: Connection Strength from excitatory to Inhibitory Node
        :param store_recurrent_acts: If true store, recurrent X and Y activations per iteration.
                  Only for debugging
        """
        super(CurrentDivisiveInhibitLayer, self).__init__()

        self.lateral_e_size = lateral_e_size
        self.lateral_i_size = lateral_i_size
        self.edge_out_ch = edge_out_ch
        self.n_iters = n_iters
        self.use_recurrent_batch_norm = use_recurrent_batch_norm

        # Only for Debugging
        self.store_recurrent_acts = store_recurrent_acts
        if self.store_recurrent_acts:
            self.x_per_iteration = []
            self.y_per_iteration = []

        # Parameters
        if a is not None:
            self.a = nn.Parameter(torch.ones(edge_out_ch) * a, requires_grad=False)
        else:
            self.a = nn.Parameter(torch.rand(edge_out_ch), requires_grad=True)  # RV between [0, 1]

        if b is not None:
            self.b = nn.Parameter(torch.ones(edge_out_ch) * b, requires_grad=False)
        else:
            self.b = nn.Parameter(torch.rand(edge_out_ch), requires_grad=True)  # RV between [0, 1]

        # Remove self excitation, a form is already included in the lateral connections
        # self.j_xx = nn.Parameter(torch.rand(edge_out_ch))
        # init.xavier_normal_(self.j_xx.view(1, edge_out_ch))

        if j_xy is not None:
            self.j_xy = nn.Parameter(torch.ones(edge_out_ch) * j_xy, requires_grad=False)
        else:
            # self.j_xy = nn.Parameter(torch.rand(edge_out_ch), requires_grad=True)
            # init.xavier_normal_(self.j_xy.view(1, edge_out_ch))

            # Empirically found this initialization, results in good IoU and gain curves
            self.j_xy = nn.Parameter(torch.ones(edge_out_ch) * -2.2, requires_grad=True)

        if j_yx is not None:
            self.j_yx = nn.Parameter(torch.ones(edge_out_ch) * j_yx, requires_grad=False)
        else:
            # self.j_yx = nn.Parameter(torch.rand(edge_out_ch), requires_grad=True)
            # init.xavier_normal_(self.j_yx.view(1, edge_out_ch))

            # Empirically found this initialization, results in good IoU and gain curves
            self.j_yx = nn.Parameter(torch.ones(edge_out_ch)*-2.2, requires_grad=True)

        self.e_bias = nn.Parameter(torch.ones(edge_out_ch)*0.01, requires_grad=True)
        self.i_bias = nn.Parameter(torch.ones(edge_out_ch)*0.01, requires_grad=True)

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

        if use_recurrent_batch_norm:
            self.recurrent_e_BN = nn.ModuleList([])
            self.recurrent_i_BN = nn.ModuleList([])
            for i in range(self.n_iters):
                self.recurrent_e_BN.append(torch.nn.BatchNorm2d(num_features=self.edge_out_ch))
                self.recurrent_i_BN.append(torch.nn.BatchNorm2d(num_features=self.edge_out_ch))

            # # Recurrent batch norm as defined in [Coolijmans et. al. - 2017]
            # # Single set of alpha\beta but different tracked running mean and var per iteration
            # self.recurrent_BN_e = RecurrentBatchNorm(self.edge_out_ch, self.n_iters)
            # self.recurrent_BN_i = RecurrentBatchNorm(self.edge_out_ch, self.n_iters)

            # # Initialization of Batch normalization layer as defined in  [Coolijmans et. al. - 2017]
            # # No bias and alpha init to 0.1
            # self.recurrent_BN_e.weight.data.fill_(0.1)
            # self.recurrent_BN_e.bias.data.fill_(0)
            # self.recurrent_BN_e.bias.requires_grad = False
            #
            # self.recurrent_BN_i.weight.data.fill_(0.1)
            # self.recurrent_BN_i.bias.data.fill_(0)
            # self.recurrent_BN_i.bias.requires_grad = False

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
                    ff +
                    self.e_bias.view(1, self.edge_out_ch, 1, 1) * torch.ones_like(ff) +
                    nn.functional.relu(self.lateral_e(f_x))
                ) / (1 + (sigmoid_j_xy * f_y))

            y = (1 - gated_b) * y + \
                gated_b * (
                    (sigmoid_j_yx * f_x) +
                    self.i_bias.view(1, self.edge_out_ch, 1, 1) * torch.ones_like(ff) +
                    nn.functional.relu(self.lateral_i(f_x))
                )

            if self.use_recurrent_batch_norm:
                f_x = nn.functional.relu(self.recurrent_e_BN[i](x))
                f_y = nn.functional.relu(self.recurrent_i_BN[i](y))
            else:
                f_x = nn.functional.relu(x)
                f_y = nn.functional.relu(y)
                # f_x = threshold_relu(x, thres=0.1, below_value=0, above_thres_multiplier=1)
                # f_y = threshold_relu(y, thres=0.1, below_value=0, above_thres_multiplier=2)

            if self.store_recurrent_acts:
                self.x_per_iteration.append(x)
                self.y_per_iteration.append(y)

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

        # Extra layers between edge extract and contour integration layer. Improves performance
        x = self.bn1(x)
        x = nn.functional.relu(x)

        x = self.contour_integration_layer(x)

        x = self.classifier(x)

        return x


class ContourIntegrationResnet50(nn.Module):
    def __init__(self, contour_integration_layer, pre_trained_edge_extract=True, classifier=ClassifierHead):
        """
        Build edge extract + contour integration layer.

        Edge extract layer is the first convolutional layer of a ResNet50. The specified contour
        integration layer is added on top of it.

        Between Edge Extract and the contour integration layers, an additional Batch Normalization
        and a max pooling layer are added.

        :param contour_integration_layer:
        :param pre_trained_edge_extract: if True [default], use pretrained Edge Extract Layer
        :param classifier: What is added on top the contour integration Layer [default = ClassifierHead]
        """
        super(ContourIntegrationResnet50, self).__init__()

        self.pre_trained_edge_extract = pre_trained_edge_extract

        self.edge_extract = torchvision.models.resnet50(pretrained=self.pre_trained_edge_extract).conv1
        if self.pre_trained_edge_extract:
            self.edge_extract.weight.requires_grad = False
        else:
            init.xavier_normal_(self.edge_extract.weight)

        # Additional Layers - see forward function for rational.
        self.num_edge_extract_chan = self.edge_extract.weight.shape[0]
        self.bn1 = nn.BatchNorm2d(num_features=self.num_edge_extract_chan)
        self.max_pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.contour_integration_layer = contour_integration_layer

        self.classifier = classifier(self.num_edge_extract_chan)

    def forward(self, in_img):
        # Edge Extraction
        x = self.edge_extract(in_img)

        # Additional Layers
        # The following layers, inserted between the edge extract and the contour integration layer,
        # are directly from ResNet50. Attaching the contour integration layer here (as opposed to
        # directly after the conv layer) gives the same dimensions as after the first layer of AlexNet
        # This allows to use the same classification head.
        x = self.bn1(x)
        x = nn.functional.relu(x)
        x = self.max_pool1(x)

        x = self.contour_integration_layer(x)

        x = self.classifier(x)

        return x


def embed_into_resnet50(edge_extract_and_contour_integration_layers, pretrained=True):
    """
    Returns a full ResNet50 Model with a contour integration layer embedded in it

    :param edge_extract_and_contour_integration_layers:
    :param pretrained: Whether the reset of the Model (everything after the contour integration layer)
       should be loaded with pretrained ImageNet Weights

    :return: ResNet50 model with embedded contour integration Layer
    """

    model = torchvision.models.resnet50(pretrained=pretrained)

    # ContourIntegrationResnet50() attaches after the first max pooling layer of ResNet50.
    # Replace one layer of Resnet50 with the cont_int_model and force all others
    # to be dummy Heads (pass through as is)
    model.conv1 = edge_extract_and_contour_integration_layers
    model.bn1 = DummyHead()
    model.relu = DummyHead()
    model.maxpool = DummyHead()

    return model


def embed_into_alexnet(edge_extract_and_contour_integration_layers, pretrained=True):
    """
    Returns a full AlexNet Model with a contour integration layer embedded in it

    :param edge_extract_and_contour_integration_layers:
    :param pretrained: Whether the reset of the Model (everything after the contour integration layer)
       should be loaded with pretrained ImageNet Weights

    :return: ResNet50 model with embedded contour integration Layer
    """

    model = torchvision.models.alexnet(pretrained=pretrained)

    # ContourIntegrationResnet50() attaches after the first max pooling layer of ResNet50.
    # Replace one layer of Resnet50 with the cont_int_model and force all others
    # to be dummy Heads (pass through as is)
    model.features[0] = edge_extract_and_contour_integration_layers

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


class BinaryClassifierResnet50(nn.Module):
    """
          Model for  Binary Classification on Images
    """
    def __init__(self, contour_integration_layer, pre_trained_edge_extract=True):
        super(BinaryClassifierResnet50, self).__init__()

        self.pre_trained_edge_extract = pre_trained_edge_extract

        self.edge_extract = torchvision.models.resnet50(
            pretrained=self.pre_trained_edge_extract).conv1
        if self.pre_trained_edge_extract:
            self.edge_extract.weight.requires_grad = False
        else:
            init.xavier_normal_(self.edge_extract.weight)

        self.num_edge_extract_chan = self.edge_extract.weight.shape[0]
        self.bn1 = nn.BatchNorm2d(num_features=self.num_edge_extract_chan)

        # maxpool layer is directly from Resnet50. Attaching the contour integration
        # layer here (as opposed to after the conv layer) since this has the same dimensions as
        # alexnet edge extract. Allows to use the same classification head (Contour task)
        # NOTE that this max pooling layer was not originally included in the pathfinder model
        self.max_pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.contour_integration_layer = contour_integration_layer

        self.classifier = BinaryClassifier(n_in_channels=self.num_edge_extract_chan)

    def forward(self, in_img):
        # Edge Extraction
        x = self.edge_extract(in_img)
        x = self.bn1(x)
        x = nn.functional.relu(x)
        x = self.max_pool1(x)

        x = self.contour_integration_layer(x)

        x = self.classifier(x)

        return x


class JointPathfinderContourResnet50(nn.Module):
    """
    Model for training on both the contour and pathfinder dataset.
    The contour integration layer is common, but separate heads for used for
    contour and pathfinder outputs.
    """
    def __init__(self, contour_integration_layer, pre_trained_edge_extract=True):
        super(JointPathfinderContourResnet50, self).__init__()

        self.pre_trained_edge_extract = pre_trained_edge_extract

        self.edge_extract = torchvision.models.resnet50(
            pretrained=self.pre_trained_edge_extract).conv1
        if self.pre_trained_edge_extract:
            self.edge_extract.weight.requires_grad = False
        else:
            torch.nn.init.xavier_normal_(self.edge_extract.weight)

        self.num_edge_extract_chan = self.edge_extract.weight.shape[0]
        self.bn1 = nn.BatchNorm2d(num_features=self.num_edge_extract_chan)

        # maxpool layer is directly from Resnet50. Attaching the contour integration
        # layer here (as opposed to after the conv layer) since this has the same dimensions as
        # alexnet edge extract. Allows to use the same classification head (Contour task)
        # NOTE that this max pooling layer was not originally included in the pathfinder model
        self.max_pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.contour_integration_layer = contour_integration_layer

        self.pathfinder_classifier = BinaryClassifier(n_in_channels=self.num_edge_extract_chan)

        self.contour_classifier = ClassifierHead(num_channels=self.num_edge_extract_chan)

    def forward(self, in_img):

        # Edge Extraction
        x = self.edge_extract(in_img)
        x = self.bn1(x)
        x = nn.functional.relu(x)
        x = self.max_pool1(x)

        x = self.contour_integration_layer(x)

        out_contour = self.contour_classifier(x)
        out_pathfinder = self.pathfinder_classifier(x)

        return out_contour, out_pathfinder
