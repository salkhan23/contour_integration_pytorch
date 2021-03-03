# -------------------------------------------------------------------------------------------------
#  Find the parameters of the best fit 2D Gabor Filter for a given L1 layer kernel. Bet fit Gabors
#  for each channel are found independently.
#
# Author: Salman Khan
# Date  : 17/09/17
# -------------------------------------------------------------------------------------------------
from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as optimize
import copy


def gabor_2d(inputs, x0, y0, theta_deg, amp, sigma, lambda1, psi, gamma):
    """
    2D Spatial Gabor Filter (Real Component only).

    Ref: [1] J. Movellan - 2002 - Tutorial on Gabor Filters
         [2] https://en.wikipedia.org/wiki/Gabor_filter

    Note: Compared to the definitions in the references above. The theta (orientation of the gaussian envelope)
    angle is reversed (compare x_prime and y_prime definitions with reference). In the references, theta rotates
    in the clockwise direction. Here we change it to rotate in the more conventional counter clockwise direction
    (theta=0, corresponds to the x axis.). Note also that this is the angle of the gaussian envelop with
    is orthogonal to the direction of the stripes.

    :param inputs: (tuple of x, y) to consider
    :param x0: x-coordinate of center of Gaussian
    :param y0: y-coordinate of center of Gaussian
    :param theta_deg: Orientation of the Gaussian or the orientation of the normal to the sinusoid (carrier)
        component of the Gabor. It is specified in degrees to allow curve_fit greater resolution when finding the
        optimal parameters
    :param amp: Amplitude of the Gabor Filter
    :param sigma: width of the Gaussian (envelope) component
    :param lambda1: Wavelength of the sinusoid (carrier) component
    :param psi: phase offset of the sinusoid component
    :param gamma: Scale ratio of the x vs y spatial extent of the Gaussian envelope.

    :return: 1D vector of 2D spatial gabor function over (x, y). Note it needs to be reshaped to get the 2D
        version. It is done this way because curve fit function, expects a single vector of inputs to optimize over
    """
    x = inputs[0]
    y = inputs[1]

    sigma = np.float(sigma)

    theta = theta_deg * np.pi / 180.0

    x_prime = (x - x0) * np.cos(theta) - (y - y0) * np.sin(theta)
    y_prime = (x - x0) * np.sin(theta) + (y - y0) * np.cos(theta)

    out = amp * np.exp(-(x_prime ** 2 + (gamma ** 2 * y_prime ** 2)) / (2 * sigma ** 2)) * \
        np.cos(2 * np.pi * x_prime / lambda1 + psi)

    # print(x0, y0, theta_deg, amp, sigma, lambda1, psi, gamma)

    return out.ravel()


def get_gabor_fragment(g_params, spatial_size):
    """
    Constructed a 2D Gabor fragment from the specified params of the specified size.
    A 3 channel fragment is always generated.

    params is  either a dictionary of a list of dictionaries of the type specified below.

    params = {
        'x0': x0,
        'y0': y0,
        'theta_deg': theta_deg,
        'amp': amp,
        'sigma': sigma,
        'lambda1': lambda1,
        'psi': psi,
        'gamma': gamma
    }

    If a single dictionary is specified all three channels have the same parameters, else if
    a list of size 3 is specified each channel has its own parameters.

    :param g_params: is either a dictionary of a list of dictionaries
    :param spatial_size:
    :return:
    """
    half_x = spatial_size[0] // 2
    half_y = spatial_size[1] // 2

    x = np.linspace(-half_x, half_x, spatial_size[0])
    y = np.linspace(-half_y, half_y, spatial_size[1])

    xx, yy = np.meshgrid(x, y)

    if type(g_params) is list and len(g_params) not in (1, 3):
        raise Exception("Only length 3 list of parameters can be specified")

    if type(g_params) is not list:

        frag = gabor_2d(
            (xx, yy),
            x0=g_params['x0'],
            y0=g_params['y0'],
            theta_deg=g_params['theta_deg'],
            amp=g_params['amp'],
            sigma=g_params['sigma'],
            lambda1=g_params['lambda1'],
            psi=g_params['psi'],
            gamma=g_params['gamma']
        )

        frag = frag.reshape((x.shape[0], y.shape[0]))
        frag = np.stack((frag, frag, frag), axis=2)
    else:

        frag = np.zeros((spatial_size[0], spatial_size[1], 3))

        for c_idx, c_params in enumerate(g_params):
            frag_chan = gabor_2d(
                (xx, yy),
                x0=c_params['x0'],
                y0=c_params['y0'],
                theta_deg=c_params['theta_deg'],
                amp=c_params['amp'],
                sigma=c_params['sigma'],
                lambda1=c_params['lambda1'],
                psi=c_params['psi'],
                gamma=c_params['gamma']
            )

            frag_chan = frag_chan.reshape((x.shape[0], y.shape[0]))
            frag[:, :, c_idx] = frag_chan

            if len(g_params) == 1:  # if length 1, set all channels the same
                frag[:, :, 1] = frag_chan
                frag[:, :, 2] = frag_chan

    # Normalize to range 0 - 255
    frag = (frag - frag.min()) / (frag.max() - frag.min()) * 255

    frag = frag.astype(np.uint8)

    return frag


def find_best_fit_2d_gabor(kernel, verbose=0):
    """
    Find the best fit parameters of a 2D gabor for each input channel of kernel.
    Channel = Last index

    :param kernel: Alexnet l1 kernel
    :param verbose: Controls verbosity of prints (
        0=Nothing is printed[Default],
        1=print optimal params,
        2=print goodness of fits)

    :return: list of best fit parameters for each channel of kernel. Format: [x, y, chan]
    """
    n_channels = kernel.shape[-1]

    half_x = kernel.shape[0] // 2
    half_y = kernel.shape[1] // 2

    x = np.linspace(-half_x, half_x, kernel.shape[0])
    y = np.linspace(-half_y, half_y, kernel.shape[1])

    xx, yy = np.meshgrid(x, y)

    opt_params_list = []

    for c_idx in range(n_channels):

        opt_params_found = False

        theta = -90

        # gabor_2d(     x0,      y0, theta_deg,     amp, sigma, lambda1,       psi, gamma):
        bounds = ([-half_x, -half_y,      -90,     -2,   0.1,       0,   -half_x,     0],
                  [ half_x,  half_y,       89,      2,     4,      20,    half_x,     2])

        while not opt_params_found:

            p0 = [0, 0, theta, 1, 2.5, 8, 0, 1]
            # p0 = [0, 0, theta, -1, 1, 8, 0, 1] # Better for black on white gabors

            try:
                popt, pcov = optimize.curve_fit(
                    gabor_2d, (xx, yy), kernel[:, :, c_idx].ravel(), p0=p0, bounds=bounds)

                # 1 SD of error in estimate
                one_sd_error = np.sqrt(np.diag(pcov))

                # Check that error in the estimate is reasonable
                if one_sd_error[2] <= 3.0:

                    opt_params_found = True
                    opt_params_list.append(popt)

                    if verbose > 0:
                        print("chan {0}: (x0,y0)=({1:0.2f},{2:0.2f}), theta={3:0.2f}, A={4:0.2f}, sigma={5:0.2f}, "
                              "lambda={6:0.2f}, psi={7:0.2f}, gamma={8:0.2f}".format(
                               c_idx, popt[0], popt[1], popt[2], popt[3], popt[4], popt[5], popt[6], popt[7]))

                    if verbose > 1:
                        print("1SD Err : (x0,y0)=({0:0.2f},{1:0.2f}), theta={2:0.2f}, A={3:0.2f}, sigma={4:0.2f}, "
                              "lambda={5:0.2f}, psi={6:0.2f}, gamma={7:0.2f}".format(
                               one_sd_error[0], one_sd_error[1], one_sd_error[2], one_sd_error[3], one_sd_error[4],
                               one_sd_error[5], one_sd_error[6], one_sd_error[7]))

                else:
                    theta += 10

            except RuntimeError:
                theta += 10

                if theta == 90:
                    # print("Optimal parameters could not be found")
                    opt_params_found = True
                    opt_params_list.append(None)

            except ValueError:
                # print("Optimal parameters could not be found")
                opt_params_found = True
                opt_params_list.append(None)

    return opt_params_list


def plot_kernel_and_best_fit_gabors(kernel, kernel_idx, fitted_gabors_params):
    """

    :param kernel:
    :param kernel_idx: Index of the kernel (only fir title)
    :param fitted_gabors_params: list of fitted parameters for each channel of kernel

    :return: None
    """
    n_channels = kernel.shape[-1]

    x_arr = np.arange(-0.5, 0.5, 1 / np.float(kernel.shape[0]))
    y_arr = np.copy(x_arr)
    xx, yy = np.meshgrid(x_arr, y_arr)

    # Higher resolution display
    x2_arr = np.arange(-0.5, 0.5, 1 / np.float((kernel.shape[0]) + 100))
    y2_arr = np.copy(x2_arr)
    xx2, yy2 = np.meshgrid(x2_arr, y2_arr)

    f = plt.figure()

    # Normalize the kernel to [0, 1] to display it properly
    display_kernel = (kernel - kernel.min()) / (kernel.max() - kernel.min())

    for c_idx in range(n_channels):

        # Plot the kernel
        f.add_subplot(n_channels, 3, (c_idx * 3) + 1)
        plt.imshow(display_kernel[:, :, c_idx], cmap='seismic')
        plt.title(r"$Chan=%d $" % c_idx)

        if np.any(fitted_gabors_params[c_idx]):  # if it is not none

            x0, y0, theta, amp, sigma, lambda1, psi, gamma = fitted_gabors_params[c_idx]

            # Fitted gabor - same resolution (with which fit was done)
            f.add_subplot(n_channels, 3, (c_idx * 3) + 2)
            fitted_gabor = gabor_2d((xx, yy), x0, y0, theta, amp, sigma, lambda1, psi, gamma)
            fitted_gabor = fitted_gabor.reshape((x_arr.shape[0], y_arr.shape[0]))
            display_gabor = (fitted_gabor - fitted_gabor.min()) / (fitted_gabor.max() - fitted_gabor.min())
            plt.imshow(display_gabor, cmap='seismic')

            # # Fitted gabor - higher resolution
            f.add_subplot(n_channels, 3, (c_idx * 3) + 3)
            fitted_gabor = gabor_2d((xx2, yy2), x0, y0, theta, amp, sigma, lambda1, psi, gamma)
            fitted_gabor = fitted_gabor.reshape((x2_arr.shape[0], y2_arr.shape[0]))
            display_gabor = (fitted_gabor - fitted_gabor.min()) / (fitted_gabor.max() - fitted_gabor.min())
            plt.imshow(display_gabor, cmap='seismic')

    f.suptitle("2D Gabor Fits for L1 Filter @ Index %d" % kernel_idx)


def get_l1_filter_orientation_and_offset(tgt_filt, tgt_filt_idx, show_plots=True):
    """
    Given a Target AlexNet L1 Convolutional Filter, fit to a 2D spatial Gabor. Use this to as
    the orientation of the filter and calculate the row shift offset to use when tiling fragments.
    This offset represents the shift in pixels to use for each row  as you move away from the
    center row. Thereby allowing contours for the target filter to be generated.

    Raises an exception if no best fit parameters are found for any of the channels of the target
    filter.

    :param show_plots:
    :param tgt_filt_idx:
    :param tgt_filt:

    :return: optimal orientation, row offset.
    """
    tgt_filt_len = tgt_filt.shape[0]

    best_fit_params_list = find_best_fit_2d_gabor(tgt_filt)

    # Plot the best fit params
    if show_plots:
        plot_kernel_and_best_fit_gabors(tgt_filt, tgt_filt_idx, best_fit_params_list)

    # Remove all empty entries
    best_fit_params_list = [c_params for c_params in best_fit_params_list if c_params is not None]
    if not best_fit_params_list:
        # raise Exception("Optimal Params could not be found")
        return np.NaN, np.NaN

    # Find channel with highest energy (Amplitude) and use its preferred orientation
    # Best fit parameters: x0, y0, theta, amp, sigma, lambda1, psi, gamma
    best_fit_params_list = np.array(best_fit_params_list)
    amp_arr = best_fit_params_list[:, 3]
    amp_arr = np.abs(amp_arr)
    max_amp_idx = np.argmax(amp_arr)

    theta_opt = best_fit_params_list[max_amp_idx, 2]

    # TODO: Fix me - Explain why this is working
    # TODO: How to handle horizontal (90) angles
    # # Convert the orientation angle into a y-offset to be used when tiling fragments
    contour_angle = theta_opt + 90.0  # orientation is of the Gaussian envelope with is orthogonal to
    # # sinusoid carrier we are interested in.
    # contour_angle = np.mod(contour_angle, 180.0)

    # if contour_angle >= 89:
    #     contour_angle -= 180  # within the defined range of tan

    # contour_angle = contour_angle * np.pi / 180.0
    # offset = np.int(np.ceil(tgt_filter_len / np.tan(contour_angle)))
    row_offset = np.int(np.ceil(tgt_filt_len / np.tan(np.pi - contour_angle * np.pi / 180.0)))

    # print("L1 kernel %d, optimal orientation %0.2f(degrees), vertical offset of tiles %d"
    #       % (tgt_filt_idx, theta_opt, row_offset))

    return theta_opt, row_offset


def get_filter_orientation(tgt_filt, o_type='average', display_params=True):
    """
    Fit the target filter to a gabor filter and find the orientation of each channel.
    If type=average, the average orientation across the filter is returned, else if
    type=max, the orientation of channel with the maximum amplitude is returned

    :param tgt_filt:
    :param o_type: ['average', 'max']
    :param display_params:

    :return: orientation of the type specified
    """
    orient = None

    gabor_fit_params = find_best_fit_2d_gabor(tgt_filt)
    gabor_fit_params = np.array(gabor_fit_params)

    if display_params:
        for c_idx, p in enumerate(gabor_fit_params):
            if p is not None:
                print("Chan {0}: (x0,y0)=({1:0.2f},{2:0.2f}), theta_deg={3:0.1f}, A={4:0.2f}, sigma={5:0.2f}, "
                      "lambda={6:0.2f}, psi={7:0.2f}, gamma={8:0.2f}".format(
                       c_idx, p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7]))
            else:
                print("Chan {0}: None")

    # Remove 'None' Entries
    gabor_fit_params_clean = [p for p in gabor_fit_params if p is not None]
    gabor_fit_params_clean = np.array(gabor_fit_params_clean)

    if len(gabor_fit_params_clean):
        o_type = o_type.lower()
        if o_type == 'average':
            orient_arr = gabor_fit_params_clean[:, 2]
            orient = np.mean(orient_arr)
        elif o_type == 'max':
            amp_arr = gabor_fit_params_clean[:, 3]
            orientation_idx = np.argmax(abs(amp_arr))
            orient = gabor_fit_params_clean[orientation_idx, 2]
        else:
            raise Exception("Unknown o_type!")

    # print(orient)
    return orient


def plot_fragment_rotations(frag, frag_params, delta_rot=15):
    """
    Plot all possible rotations (multiples of  delta_rot) of the specified fragment

    :param frag:
    :param frag_params:  Frag_params is either a dictionary or a list of dictionaries, one for each channel.
                         Format of dictionary is specified above.
    :param delta_rot:
    :return: None
    """
    if type(frag_params) is not list:
        frag_params = [frag_params]

    rotated_frag_params_list = copy.deepcopy(frag_params)

    rot_ang_arr = np.arange(0, 360, delta_rot)
    n_rows = np.int(np.floor(np.sqrt(rot_ang_arr.shape[0])))
    n_cols = np.int(np.ceil(rot_ang_arr.shape[0] / n_rows))

    f, ax_arr = plt.subplots(n_rows, n_cols)
    f.suptitle("Rotations")

    for idx, rot_ang in enumerate(rot_ang_arr):

        for c_idx, rot_frag_params in enumerate(rotated_frag_params_list):
            rot_frag_params["theta_deg"] = rot_ang + frag_params[c_idx]['theta_deg']

            if rot_frag_params["theta_deg"] > 180:
                rot_frag_params["theta_deg"] -= 360
            if rot_frag_params["theta_deg"] < -180:
                rot_frag_params["theta_deg"] += 360

            rot_frag = get_gabor_fragment(rotated_frag_params_list, frag.shape[0:2])

            row_idx = np.int(idx / n_cols)
            col_idx = idx - row_idx * n_cols
            # print(row_idx, col_idx)

            ax_arr[row_idx][col_idx].imshow(rot_frag)
            ax_arr[row_idx][col_idx].set_title("Angle = {}".format(rot_ang))
            ax_arr[row_idx][col_idx].set_xticks([])
            ax_arr[row_idx][col_idx].set_yticks([])


def convert_gabor_params_list_to_dict(params_list):
    """
    from [x0, y0, theta_deg, amp, sigma, lambda1, psi, gamma] to
    {
        'x0': x0,
        'y0': y0,
        'theta_deg': theta_deg,
        'amp': amp,
        'sigma': sigma,
        'lambda1': lambda1,
        'psi': psi,
        'gamma': gamma
    }

    :param params_list:
    :return:
    """
    g_params = []
    for ch_idx, ch_par in enumerate(params_list):
        g_params.append({
            'x0': ch_par[0],
            'y0': ch_par[1],
            'theta_deg': ch_par[2],
            'amp': ch_par[3],
            'sigma': ch_par[4],
            'lambda1': ch_par[5],
            'psi': ch_par[6],
            'gamma': ch_par[7]
        })

    return g_params


if __name__ == "__main__":
    # -----------------------------------------------------------------------------------
    # Initialization
    # -----------------------------------------------------------------------------------
    target_fragment_size = (11, 11)

    # Immutable
    plt.ion()

    # -----------------------------------------------------------------------------------
    # Sample Gabor Fragment
    # -----------------------------------------------------------------------------------
    # Single channel
    gabor_parameters = {
        'x0': 0,
        'y0': 0,
        'theta_deg': 90,
        'amp': 1,
        'sigma': 2.0,
        'lambda1': 6,
        'psi': 0,
        'gamma': 1
    }

    # # 3 channels
    # gabor_parameters = [
    #     {
    #         'x0': 0,
    #         'y0': 0,
    #         'theta_deg': 90,
    #         'amp': 1,
    #         'sigma': 2.0,
    #         'lambda1': 6,
    #         'psi': 0,
    #         'gamma': 1
    #     },
    #     {
    #         'x0': 0,
    #         'y0': 0,
    #         'theta_deg': 45,
    #         'amp': 1,
    #         'sigma': 2.0,
    #         'lambda1': 15,
    #         'psi': 0,
    #         'gamma': 1
    #     },
    #     {
    #         'x0': 0,
    #         'y0': 0,
    #         'theta_deg': 90,
    #         'amp': 1,
    #         'sigma': 2.0,
    #         'lambda1': 6,
    #         'psi': 0,
    #         'gamma': 1
    #     }
    # ]

    fragment = get_gabor_fragment(gabor_parameters, target_fragment_size)

    plt.figure()
    plt.imshow(fragment)
    plt.title("Generated Fragment")

    # -----------------------------------------------------------------------------------
    # TODO: Fit Gabor Fragment and match settings
    # -----------------------------------------------------------------------------------
