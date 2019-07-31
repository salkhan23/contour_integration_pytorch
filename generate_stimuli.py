# ---------------------------------------------------------------------------------------
# Generate Embedded Contour Stimuli similar to
# Field, Hayes & Hess - 1993 - Contour Integration by the Human Visual System: Evidence
# for a local association field "
#
# This is deep learning library independent
# ---------------------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
import gabor_fits
import copy
from skimage.transform import rotate


def get_2d_gaussian_kernel(shape, sigma=1.0):
    """
    Returns a 2d (un-normalized) Gaussian kernel of the specified shape.

    :param shape: x,y dimensions of the gaussian
    :param sigma: standard deviation of generated Gaussian
    :return:
    """
    ax = np.linspace(-1, 1, shape[0])
    ay = np.linspace(-1, 1, shape[1])

    xx, yy = np.meshgrid(ax, ay)
    kernel = np.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))

    kernel = kernel.reshape(shape)

    return kernel


def do_tiles_overlap(l1, r1, l2, r2, border_can_overlap=True):
    """
    Rectangles are specified by two points, the (x,y) coordinates of the top left corner (l1)
    and bottom right corner

    Two rectangles do not overlap if one of the following conditions is true.
    1) One rectangle is above top edge of other rectangle.
    2) One rectangle is on left side of left edge of other rectangle.

    Ref:  https://www.geeksforgeeks.org/find-two-rectangles-overlap/

    Different from Ref, and more aligned with the coordinates system in the rest of the file, x
    controls vertical while y controls horizontal

    :param border_can_overlap:
    :param l1: top left corner of tile 1
    :param r1: bottom right corner of tile 1
    :param l2:
    :param r2:

    :return:  True of the input tiles overlap, false otherwise
    """
    # Does one square lie to the Left of the other
    if border_can_overlap:
        if l1[1] >= r2[1] or l2[1] >= r1[1]:
            return False
    else:
        if l1[1] > r2[1] or l2[1] > r1[1]:
            return False

    # Does one square lie above the other
    if border_can_overlap:
        if l1[0] >= r2[0] or l2[0] >= r1[0]:
            return False
    else:
        if l1[0] > r2[0] or l2[0] > r1[0]:
            return False

    return True


def randomly_rotate_tile(tile, delta_rotation=45.0):
    """
    randomly rotate tile by 360/delta_rotation permutations

    :param delta_rotation: Angle in degrees, of which the rotated tile is a factor of
    :param tile: 2d contour fragment

    :return: rotated tile. Note this is an RGB format and values range b/w [0, 255]
    """
    num_possible_rotations = 360 // delta_rotation
    return rotate(tile, angle=(np.random.randint(0, np.int(num_possible_rotations)) * delta_rotation))


def tile_image(img, frag, insert_loc_arr, rotate_frags=True, delta_rotation=45, gaussian_smoothing=True, sigma=4.0):
    """
    Place tile 'fragments' at the specified starting positions (x, y) in the image.

    :param frag: contour fragment to be inserted
    :param insert_loc_arr: array of (x,y) position where tiles should be inserted
    :param img: image where tiles will be placed
    :param rotate_frags: If true each tile is randomly rotated before insertion.
    :param delta_rotation: min rotation value
    :param gaussian_smoothing: If True, each fragment is multiplied with a Gaussian smoothing
            mask to prevent tile edges becoming part of stimuli [they will lie in the center of the RF of
            many neurons. [Default=True]
    :param sigma: Standard deviation of gaussian smoothing mask. Only used if gaussian smoothing is True

    :return: tiled image
    """
    tile_len = frag.shape[0]

    if insert_loc_arr.ndim == 1:
        x_arr = [insert_loc_arr[0]]
        y_arr = [insert_loc_arr[1]]
    else:
        x_arr = insert_loc_arr[:, 0]
        y_arr = insert_loc_arr[:, 1]

    for idx in range(len(x_arr)):

        # print("Processing Fragment @ (%d,%d)" % (x_arr[idx], y_arr[idx]))

        if (-tile_len < x_arr[idx] < img.shape[0]) and (-tile_len < y_arr[idx] < img.shape[1]):

            start_x_loc = np.int(max(x_arr[idx], 0))
            stop_x_loc = np.int(min(x_arr[idx] + tile_len, img.shape[0] - 1))

            start_y_loc = np.int(max(y_arr[idx], 0))
            stop_y_loc = np.int(min(y_arr[idx] + tile_len, img.shape[1] - 1))

            # print("Placing Fragment at location  l1=(%d, %d), y = (%d, %d),"
            #       % (start_x_loc, stop_x_loc, start_y_loc, stop_y_loc))

            # Adjust incomplete beginning tiles
            if x_arr[idx] < 0:
                tile_x_start = tile_len - (stop_x_loc - start_x_loc)
            else:
                tile_x_start = 0

            if y_arr[idx] < 0:
                tile_y_start = tile_len - (stop_y_loc - start_y_loc)
            else:
                tile_y_start = 0
            #
            # print("Tile indices x = (%d,%d), y = (%d, %d)" % (
            #       tile_x_start, tile_x_start + stop_x_loc - start_x_loc,
            #       tile_y_start, tile_y_start + stop_y_loc - start_y_loc))

            if rotate_frags:
                tile = randomly_rotate_tile(frag, delta_rotation)
            else:
                tile = frag

            # multiply the file with the gaussian smoothing filter
            # The edges between the tiles will lie within the stimuli of some neurons.
            # to prevent these prom being interpreted as stimuli, gradually decrease them.
            if gaussian_smoothing:
                g_kernel = get_2d_gaussian_kernel((tile_len, tile_len), sigma=sigma)
                g_kernel = np.reshape(g_kernel, (g_kernel.shape[0], g_kernel.shape[1], 1))
                g_kernel = np.repeat(g_kernel, 3, axis=2)

                tile = tile * g_kernel

            # only add the parts of the fragments that lie within the image dimensions
            img[start_x_loc: stop_x_loc, start_y_loc: stop_y_loc, :] = \
                tile[tile_x_start: tile_x_start + stop_x_loc - start_x_loc,
                     tile_y_start: tile_y_start + stop_y_loc - start_y_loc, :]

    return img


def _add_single_side_of_contour_constant_separation(
        img, center_frag_start, frag, frag_params, c_len, beta, alpha, d, d_delta, frag_size,
        random_beta_rot=False, random_alpha_rot=True):
    """

    :param img:
    :param center_frag_start:
    :param frag:
    :param frag_params:
    :param c_len:
    :param beta:
    :param alpha:
    :param d:
    :param d_delta:
    :param frag_size:
    :param random_beta_rot: [False]
    :param random_alpha_rot:

    :return:
    """
    if type(frag_params) is not list:
        frag_params = [frag_params]

    tile_offset = np.zeros((2,), dtype=np.int)
    prev_tile_start = center_frag_start

    tile_starts = []

    acc_angle = 0

    for i in range(c_len // 2):

        if random_beta_rot:
            beta = np.random.choice((-1, 1), size=1) * beta

        acc_angle += beta
        # acc_angle = np.mod(acc_angle, 360)
        # print("fragment idx {} acc_angle {}".format(i, acc_angle))

        rotated_frag_params_list = copy.deepcopy(frag_params)

        if random_alpha_rot:
            alpha = np.random.choice((-1, 1), size=1) * alpha

        # Rotate the next fragment
        # ------------------------
        for c_params in rotated_frag_params_list:
            c_params["theta_deg"] += (acc_angle + alpha)

        rotated_frag = gabor_fits.get_gabor_fragment(rotated_frag_params_list, frag.shape[0:2])

        # Find the location of the next fragment
        # --------------------------------------
        # TODO: Should this be gabor params of the chan with the highest amplitude
        loc_angle = rotated_frag_params_list[0]['theta_deg'] - alpha

        # Note
        # [1] Origin of (x, y) top left corner
        # [2] Dim 0 increases downward direction, Dim 1 increases in the right direction
        # [3] Gabor angles are specified wrt y-axis i.e. 0 orientation is vertical. For position
        #     we need the angles to be relative to the x-axis.
        tile_offset[0] = d * np.cos(loc_angle / 180.0 * np.pi)
        tile_offset[1] = d * np.sin(loc_angle / 180.0 * np.pi)

        curr_tile_start = prev_tile_start + tile_offset
        # print("Current tile start {0}. (offsets {1}, previous {2}, loc_angle={3})".format(
        #     curr_tile_start, tile_offset, prev_tile_start, loc_angle))

        # check if the current tile overlaps with the previous tile
        # TODO: Check if current tile overlaps with ANY previous one.
        l1 = curr_tile_start
        r1 = l1 + frag_size
        l2 = prev_tile_start
        r2 = l2 + frag_size
        is_overlapping = do_tiles_overlap(l1, r1, l2, r2)

        while is_overlapping:
            print("Tile {0} overlaps with tile at location {1}".format(curr_tile_start, prev_tile_start))
            tile_offset[0] += d_delta * np.cos(loc_angle / 180.0 * np.pi)
            tile_offset[1] += d_delta * np.sin(loc_angle / 180.0 * np.pi)

            curr_tile_start = prev_tile_start + tile_offset
            print("Current tile relocated to {0}. (offsets {1})".format(curr_tile_start, tile_offset))

            l1 = curr_tile_start
            r1 = l1 + frag_size
            is_overlapping = do_tiles_overlap(l1, r1, l2, r2)

        img = tile_image(
            img,
            rotated_frag,
            curr_tile_start,
            rotate_frags=False,
            gaussian_smoothing=False
        )

        prev_tile_start = curr_tile_start
        tile_starts.append(curr_tile_start)

        # print("Tile places @ {}".format(curr_tile_start))
        # plt.figure()
        # plt.imshow(img)
        # input("Next?")

    return img, tile_starts


def add_contour_path_constant_separation(
        img, frag, frag_params, c_len, beta, alpha, d, center_frag_start=None,
        rand_inter_frag_direction_change=True, random_alpha_rot=True, base_contour='random'):
    """
    Add curved contours to the test image as added in the ref. a constant separation (d)
    is projected from the previous tile to find the location of the next tile.

    If the tile overlaps, fragment separation is increased by a factor of d // 4.

    :param img:
    :param frag:
    :param frag_params:
    :param c_len:
    :param beta:
    :param alpha:
    :param d:
    :param center_frag_start:
    :param rand_inter_frag_direction_change:
    :param random_alpha_rot:[True]
    :param base_contour: this determines the shape of the base contour. If set to sigmoid (default), the
        generated contour (2 calls to this function with d and -d) are symmetric about the origin, if set to
        circle, they are mirror symmetric about the vertical axis. This is for the case random_frag_direction
        is set to false.
    :return:
    """

    if base_contour.lower() not in ['sigmoid', 'circle', 'random']:
        raise Exception("Invalid base contour. Should be [sigmoid or circle]")

    if type(frag_params) is not list:
        frag_params = [frag_params]

    frag_size = np.array(frag.shape[0:2])

    if center_frag_start is None:
        img_size = np.array(img.shape[0:2])
        img_center = img_size // 2
        center_frag_start = img_center - (frag_size // 2)

    d_delta = d // 4

    # Add center fragment
    if alpha == 0:
        frag_from_contour_rot = 0
    else:
        if random_alpha_rot:
            frag_from_contour_rot = np.random.choice((-alpha, alpha), size=1)
        else:
            frag_from_contour_rot = alpha

    first_frag_params_list = copy.deepcopy(frag_params)

    for c_params in first_frag_params_list:
        c_params["theta_deg"] = c_params["theta_deg"] + frag_from_contour_rot

    first_frag = gabor_fits.get_gabor_fragment(first_frag_params_list, frag.shape[0:2])

    img = tile_image(
        img,
        first_frag,
        center_frag_start,
        rotate_frags=False,
        gaussian_smoothing=False,
    )
    c_tile_starts = [center_frag_start]

    img, tiles = _add_single_side_of_contour_constant_separation(
        img, center_frag_start, frag, frag_params, c_len, beta, alpha, d, d_delta, frag_size,
        random_beta_rot=rand_inter_frag_direction_change,
        random_alpha_rot=random_alpha_rot)
    c_tile_starts.extend(tiles)

    if base_contour == 'circle':
        beta = -beta
    elif base_contour == 'random':
        beta = np.random.choice((-1, 1), size=1) * beta

    img, tiles = _add_single_side_of_contour_constant_separation(
        img, center_frag_start, frag, frag_params, c_len, beta, alpha, -d, -d_delta, frag_size,
        random_beta_rot=rand_inter_frag_direction_change,
        random_alpha_rot=random_alpha_rot)
    c_tile_starts.extend(tiles)

    # ---------------------------
    c_tile_starts = np.array(c_tile_starts)

    return img, c_tile_starts


def get_non_overlapping_bg_fragment(f_tile_start, f_tile_size, c_tile_starts, c_tile_size, max_offset):
    """

    :param f_tile_size:
    :param f_tile_start:
    :param c_tile_starts:
    :param c_tile_size:
    :param max_offset:
    :return:
    """
    # print("get_non_overlapping_bg_fragment: full tile start: {}".format(f_tile_start))

    for r_idx in range(max_offset):
        for c_idx in range(max_offset):

            l1 = f_tile_start + np.array([r_idx, c_idx])  # top left corner of new bg tile
            r1 = l1 + c_tile_size   # lower right corner of new bg tile

            is_overlapping = False

            # print("Checking start location {}".format(l1))

            if (r1[0] > f_tile_start[0] + f_tile_size[0]) or (r1[1] > f_tile_start[1] + f_tile_size[1]):
                # new bg tile is outside the full tile
                continue

            for c_tile in c_tile_starts:

                l2 = c_tile  # contour tile top left corner
                r2 = c_tile + c_tile_size  # bottom right corner of new bg tile
                # print("checking bg tile @ start {0} with contour tile @ start {1}".format(l1, l2))

                if do_tiles_overlap(l1, r1, l2, r2, border_can_overlap=False):
                    # print('overlaps!')
                    is_overlapping = True
                    break

            if not is_overlapping:
                # print("Found non-overlapping")
                return l1
    return None


def get_background_tiles_locations(frag_len, img_len, row_offset, space_bw_tiles, tgt_n_visual_rf_start):
    """
    Starting locations for non-overlapping fragment tiles to cover the whole image.
    if row_offset is non-zero, tiles in each row are shifted by specified amount as you move further away
    from the center row.

    :param space_bw_tiles:
    :param tgt_n_visual_rf_start:
    :param row_offset:
    :param frag_len:
    :param img_len:

    :return: start_x, start_y
    """
    frag_spacing = np.int(frag_len + space_bw_tiles)
    n_tiles = np.int(img_len // frag_spacing)

    # To handle non-zero row shift, we have to add additional tiles, so that the whole image is populated
    add_tiles = 0
    if row_offset:
        max_shift = (n_tiles // 2 + 1) * row_offset
        add_tiles = abs(max_shift) // frag_spacing + 1
    # print("Number of tiles in image %d, number of additional tiles %d" % (n_tiles, add_tiles))

    n_tiles += add_tiles
    if n_tiles & 1 == 1:  # make even
        n_tiles += 1

    zero_offset_starts = np.arange(
        tgt_n_visual_rf_start - (n_tiles / 2) * frag_spacing,
        tgt_n_visual_rf_start + (n_tiles / 2 + 1) * frag_spacing,
        frag_spacing,
    )

    # Fist dimension stays the same
    start_x = np.repeat(zero_offset_starts, len(zero_offset_starts))

    # In the second dimension each row is shifted by offset from the center row

    # # If there is nonzero spacing between tiles, the offset needs to be updated
    if space_bw_tiles:
        row_offset = np.int(frag_spacing / np.float(frag_len) * row_offset)

    start_y = []
    for row_idx in np.arange(-n_tiles / 2, (n_tiles / 2) + 1):

        # print("processing row at idx %d, offset=%d" % (row_idx, row_idx * row_offset))

        ys = np.array(zero_offset_starts) + (row_idx * row_offset)
        start_y.append(ys)

    start_y = np.array(start_y)
    start_y = np.reshape(start_y, (start_y.shape[0] * start_y.shape[1]))

    loc_arr = np.array([start_x, start_y]).astype(int)
    loc_arr = loc_arr.T

    return loc_arr


def add_background_fragments(img, frag, c_frag_starts, f_tile_size, delta_rotation, frag_params,
                             relocate_allowed=True):
    """

    Divides the image into f(ull)_tile_size, for all locations that do not contain a contour fragment
    add a background fragment at a random location with a random orientation.

    :param img:
    :param frag:
    :param c_frag_starts:
    :param f_tile_size:
    :param delta_rotation:
    :param frag_params:
    :param relocate_allowed: If a bg frag overlaps with a contour fragment, try to
        relocate fragment, so it can fit in the tile without overlapping with the
        contour fragment

    :return: (1) image with background tiles added
             (2) array of bg fragment tiles
             (3) array of bg fragment tiles removed
             (4) array of bg fragment tiles that were relocated
    """
    img_size = np.array(img.shape[0:2])
    img_center = img_size // 2

    center_f_tile_start = img_center - f_tile_size // 2

    # Get start locations of all full tiles
    f_tile_starts = get_background_tiles_locations(
        frag_len=f_tile_size[0],
        img_len=max(img_size[0], img_size[1]),
        row_offset=0,
        space_bw_tiles=0,
        tgt_n_visual_rf_start=center_f_tile_start[0]
    )

    # Displace the stimulus fragment in each full tile
    max_displace = f_tile_size[0] - frag.shape[0]
    bg_frag_starts = np.copy(f_tile_starts)

    if max_displace != 0:
        bg_frag_starts += np.random.randint(0, max_displace, f_tile_starts.shape)

    # Remove or replace all tiles that overlap with contour path fragments
    # --------------------------------------------------------------------
    removed_bg_frag_starts = []
    relocate_bg_frag_starts = []

    for c_frag_start in c_frag_starts:

        c_frag_start = np.expand_dims(c_frag_start, axis=0)

        # Find overlapping background fragments
        dist_to_c_frag = np.linalg.norm(c_frag_start - bg_frag_starts, axis=1)
        # for ii, dist in enumerate(dist_to_c_frag):
        #     print("{0}: {1}".format(ii, dist))

        ovlp_bg_frag_idx_arr = np.argwhere(dist_to_c_frag <= np.sqrt(2)*frag.shape[0])
        # for idx in ovlp_bg_frag_idx_arr:
        #     print("contour fragment @ {0}, overlaps with bg fragment @ index {1} and location {2}".format(
        #         c_frag_start, idx, bg_frag_starts[idx, :]))

        ovlp_bg_frag_idx_to_remove = []

        for ii, bg_frag_idx in enumerate(ovlp_bg_frag_idx_arr):

            f_tile_start = f_tile_starts[bg_frag_idx, :]

            no_ovrlp_bg_frag = None
            if relocate_allowed:
                # Is relocation possible?
                no_ovrlp_bg_frag = get_non_overlapping_bg_fragment(
                    f_tile_start=np.squeeze(f_tile_start, axis=0),
                    f_tile_size=f_tile_size,
                    c_tile_starts=c_frag_starts,
                    c_tile_size=frag.shape[0:2],
                    max_offset=max_displace,
                )

            if no_ovrlp_bg_frag is not None:
                # print("Relocating tile @ {0} to {1}".format(bg_frag_starts[bg_frag_idx, :], no_ovrlp_bg_frag))

                bg_frag_starts[bg_frag_idx, :] = np.expand_dims(no_ovrlp_bg_frag, axis=0)
                relocate_bg_frag_starts.append(no_ovrlp_bg_frag)

            else:
                # print("Remove bg fragment at index {0}, location {1}".format(
                #     bg_frag_idx, bg_frag_starts[bg_frag_idx, :]))

                removed_bg_frag_starts.append(bg_frag_starts[bg_frag_idx, :])
                ovlp_bg_frag_idx_to_remove.append(bg_frag_idx)

        # Remove the tiles that cannot be replaced from bg_frag and bg_full lists
        bg_frag_starts = \
            np.delete(bg_frag_starts, ovlp_bg_frag_idx_to_remove, axis=0)

        f_tile_starts = \
            np.delete(f_tile_starts, ovlp_bg_frag_idx_to_remove, axis=0)

    removed_bg_frag_starts = np.array(removed_bg_frag_starts)

    if removed_bg_frag_starts.size > 0:
        removed_bg_frag_starts = np.squeeze(removed_bg_frag_starts, axis=1)

    relocate_bg_frag_starts = np.array(relocate_bg_frag_starts)

    # Now add the background fragment tiles
    # -------------------------------------
    if type(frag_params) is not list:
        frag_params = [frag_params]
    rotated_frag_params_list = copy.deepcopy(frag_params)

    num_possible_rotations = 360 // delta_rotation

    for start in bg_frag_starts:

        random_rotation = np.random.randint(0, np.int(num_possible_rotations)) * delta_rotation
        for c_params in rotated_frag_params_list:
            c_params['theta_deg'] = c_params['theta_deg'] + random_rotation

        rotated_frag = gabor_fits.get_gabor_fragment(rotated_frag_params_list, frag.shape[0:2])

        img = tile_image(
            img,
            rotated_frag,
            start.astype(int),
            rotate_frags=False,
            gaussian_smoothing=False
        )

    return img, bg_frag_starts, removed_bg_frag_starts, relocate_bg_frag_starts


def highlight_tiles(in_img, tile_shape, insert_loc_arr, edge_color=(255, 0, 0)):
    """
    Highlight specified tiles in the image


    :param in_img:
    :param tile_shape:
    :param insert_loc_arr:
    :param edge_color:

    :return: output image with the tiles highlighted
    """
    out_img = np.copy(in_img)

    img_size = in_img.shape[:2]
    tile_len = tile_shape[0]

    if insert_loc_arr.ndim == 1:
        x_arr = [insert_loc_arr[0]]
        y_arr = [insert_loc_arr[1]]
    else:
        x_arr = insert_loc_arr[:, 0]
        y_arr = insert_loc_arr[:, 1]

    for idx in range(len(x_arr)):
        # print("idx {}, x={},y={}".format(idx, x_arr[idx], y_arr[idx]))

        if (-tile_len < x_arr[idx] < img_size[0]) and (-tile_len < y_arr[idx] < img_size[1]):

            start_x_loc = max(x_arr[idx], 0)
            stop_x_loc = min(x_arr[idx] + tile_len, img_size[0]-1)

            start_y_loc = max(y_arr[idx], 0)
            stop_y_loc = min(y_arr[idx] + tile_len, img_size[1]-1)

            # print("Highlight tile @ tl=({0}, {1}), br=({2},{3})".format(
            #     start_x_loc, start_y_loc, stop_x_loc, stop_y_loc))

            out_img[start_x_loc: stop_x_loc, start_y_loc, :] = edge_color
            out_img[start_x_loc: stop_x_loc, stop_y_loc, :] = edge_color

            out_img[start_x_loc, start_y_loc: stop_y_loc, :] = edge_color
            out_img[stop_x_loc, start_y_loc: stop_y_loc, :] = edge_color

    return out_img


def get_mean_pixel_value_at_boundary(frag, width=1):
    """

    :return:
    """
    x_top = frag[0:width, :, :]
    x_bottom = frag[-width:, :, :]

    y_top = frag[:, 0:width, :]
    y_bottom = frag[:, -width:, :]

    y_top = np.transpose(y_top, axes=(1, 0, 2))
    y_bottom = np.transpose(y_bottom, axes=(1, 0, 2))

    border_points = np.array([x_top, x_bottom, y_top, y_bottom])

    mean_border_value = np.mean(border_points, axis=(0, 1, 2))
    mean_border_value = [np.uint8(ch) for ch in mean_border_value]

    return mean_border_value


def generate_contour_image(
        frag, frag_params, c_len, beta, alpha, f_tile_size, img_size=None, bg_frag_relocate=True,
        rand_inter_frag_direction_change=True, random_alpha_rot=True, center_frag_start=None, base_contour='random'):
    """

    Generate image and label for specified fragment parameters.

    In [Fields -1993] a small visible stimulus is placed inside a large tile
    Here, full tile refers to the large tile & fragment tile refers to the visible stimulus. If the
    fragment is part of the contour, it is called a contour fragment otherwise a background fragment

    :param frag:
    :param frag_params:
    :param c_len:
    :param beta:
    :param alpha:
    :param f_tile_size:
    :param img_size: [Default = (227, 227, 3)]
    :param bg_frag_relocate: if a bg fragment overlaps with a contour fragment within a tile. Try to
            relocate it before removing it.
    :param rand_inter_frag_direction_change: Curve Direction changes randomly between component framgents.
            [Default =True]
    :param random_alpha_rot:
    :param center_frag_start: starting location of contour fragments
    :param base_contour:

    :return: img, label
        The labels is a numpy array of 0 or 1 identifying which full tiles contain contour fragments
    """
    if img_size is None:
        img_size = np.array([227, 227, 3])

    img_center = img_size[0:2] // 2

    frag_size = np.array(frag.shape[0:2])

    if center_frag_start is None:
        center_frag_start = img_center - (frag_size // 2)

    bg = get_mean_pixel_value_at_boundary(frag)

    # Get the full tiles for the image
    center_f_tile_start = img_center - (f_tile_size // 2)

    f_tile_starts = get_background_tiles_locations(
        frag_len=f_tile_size[0],
        img_len=img_size[0],
        row_offset=0,
        space_bw_tiles=0,
        tgt_n_visual_rf_start=center_f_tile_start[0]
    )
    f_tile_centers = f_tile_starts + (f_tile_size // 2)
    num_f_tiles = len(f_tile_starts)
    f_tiles_single_dim = np.int(np.sqrt(num_f_tiles))

    img = np.ones(img_size, dtype=np.uint8) * bg
    label = np.zeros(num_f_tiles, dtype='uint8')

    # Image  -------------------------------------
    # Get Contour Fragments
    if (c_len > 1) or (c_len == 1 and beta == 0):
        img, c_frag_starts = add_contour_path_constant_separation(
            img, frag, frag_params, c_len, beta, alpha, f_tile_size[0],
            center_frag_start=center_frag_start,
            rand_inter_frag_direction_change=rand_inter_frag_direction_change,
            random_alpha_rot=random_alpha_rot,
            base_contour=base_contour
        )
    else:
        c_frag_starts = []
        # For all other cases, c_len ==1 and beta !=0, just add background tiles.
        # which move around within the full tile. No contour integration for single contour fragments

    # print("contour fragment starts:\n{}".format(c_frag_starts))

    # Add background fragments
    img, bg_frag_starts, removed_tiles, relocated_tiles = add_background_fragments(
        img, frag, c_frag_starts, f_tile_size, 10, frag_params, bg_frag_relocate)

    # Label -----------------------------------
    for c_frag_start in c_frag_starts:
        c_frag_center = c_frag_start + frag_size // 2

        dist_to_c_frag = np.linalg.norm(f_tile_centers - c_frag_center, axis=1)
        closest_full_tile_idx = np.argmin(dist_to_c_frag)

        # print("Closest Full Tile Index {}. Distance {}".format(
        #     closest_full_tile_idx, dist_to_c_frag[closest_full_tile_idx]))

        label[closest_full_tile_idx] = 1

    label = label.reshape(f_tiles_single_dim, f_tiles_single_dim)

    # # Debug ------------------------
    # # Highlight Contour tiles - Red
    # img = highlight_tiles(img, frag_size, c_frag_starts,  edge_color=(255, 0, 0))
    #
    # # Highlight Background Fragment tiles - Green
    # img = highlight_tiles(img, frag_size, bg_frag_starts, edge_color=(0, 255, 0))
    #
    # # Highlight Removed tiles - Blue
    # img = highlight_tiles(img, frag_size, removed_tiles, edge_color=(0, 0, 255))
    #
    # # Highlight Relocated tiles - Teal
    # img = highlight_tiles(img, frag_size, relocated_tiles, edge_color=(0, 255, 255))
    #
    # # highlight full tiles
    # img = highlight_tiles(img, f_tile_size, f_tile_starts, edge_color=(255, 255, 0))
    #
    # plt.figure()
    # plt.imshow(img)
    # print(label)

    return img, label


def get_contour_start_ranges(c_len, frag_orient, f_tile_size, img_size):
    """
    Get the min, max starting offsets so that the defined contours stays within the image

    :param c_len:
    :param frag_orient:
    :param f_tile_size:
    :param img_size:

    :return: two tuples: (x_min, x_max), (y_min, y_max)
    """
    x_extent = c_len * f_tile_size[0] * np.cos(frag_orient / 180. * np.pi)
    y_extent = c_len * f_tile_size[1] * np.sin(frag_orient / 180. * np.pi)

    # Contours are defined starting from the center fragment
    x_extent = np.int(x_extent / 2)
    y_extent = np.int(y_extent / 2)

    # min start is half the full tile
    min_start_x = x_extent + f_tile_size[0] // 2
    max_start_x = img_size[0] - f_tile_size[0] // 2 - x_extent

    min_start_y = y_extent + f_tile_size[1] // 2
    max_start_y = img_size[1] - f_tile_size[1] // 2 - y_extent

    # print("x start range [{}, {}]. y start ranges [{}, {}]".format(
    #     min_start_x, max_start_x, min_start_y, max_start_y))

    return (min_start_x, max_start_x), (min_start_y, max_start_y)


if __name__ == "__main__":
    # -----------------------------------------------------------------------------------
    # Initialization
    # -----------------------------------------------------------------------------------
    random_seed = 10

    fragment_size = (11, 11)
    full_tile_size = np.array([18, 18])

    image_size = np.array([227, 227, 3])

    # Immutable
    plt.ion()
    np.random.seed(random_seed)

    # Gabor Fragment
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

    fragment = gabor_fits.get_gabor_fragment(gabor_parameters, fragment_size)
    # plt.figure()
    # plt.imshow(fragment)
    # plt.title("Generated Fragment")

    # # -----------------------------------------------------------------------------------
    # #  Test Stimulus - Manual
    # # -----------------------------------------------------------------------------------
    # bg_value = get_mean_pixel_value_at_boundary(fragment)
    # test_image = np.ones(image_size, dtype=np.uint8) * bg_value
    #
    # contour_len = 9
    # beta_rotation = 15
    # alpha_rotation = 0
    #
    # #  Add the Contour Path
    # test_image, path_fragment_starts = add_contour_path_constant_separation(
    #     img=test_image,
    #     frag=fragment,
    #     frag_params=gabor_parameters,
    #     c_len=contour_len,
    #     beta=beta_rotation,
    #     alpha=alpha_rotation,
    #     d=full_tile_size[0],
    #     rand_inter_frag_direction_change=False,
    #     base_contour='sigmoid'
    # )
    #
    # plt.figure()
    # plt.imshow(test_image)
    # plt.title("Contour Fragments")
    #
    # # Add background Fragments
    # test_image, bg_tiles, bg_removed_tiles, bg_relocated_tiles = add_background_fragments(
    #     test_image,
    #     fragment,
    #     path_fragment_starts,
    #     full_tile_size,
    #     beta_rotation,
    #     gabor_parameters,
    #     relocate_allowed=True
    # )
    # plt.figure()
    # plt.imshow(test_image)
    # plt.title("Highlighted Tiles: Stimulus c_len = {}, beta = {}, alpha = {}".format(contour_len, beta_rotation,
    #                                                                                  alpha_rotation))
    # # Highlight Tiles
    # full_tile_starts = get_background_tiles_locations(
    #     frag_len=full_tile_size[0],
    #     img_len=image_size[1],
    #     row_offset=0,
    #     space_bw_tiles=0,
    #     tgt_n_visual_rf_start=image_size[0] // 2 - (full_tile_size[0] // 2)
    # )
    #
    # test_image = highlight_tiles(test_image, fragment_size, bg_tiles, edge_color=(255, 255, 0))
    # test_image = highlight_tiles(test_image, full_tile_size, full_tile_starts, edge_color=(255, 0, 0))
    # test_image = highlight_tiles(test_image, fragment_size, path_fragment_starts, edge_color=(0, 0, 255))
    #
    # plt.figure()
    # plt.imshow(test_image)
    # plt.title("Highlighted Tiles".format(contour_len, beta_rotation, alpha_rotation))

    # -----------------------------------------------------------------------------------
    #  Stimulus - Single function
    # -----------------------------------------------------------------------------------
    contour_len = 9
    beta_rotation = 15
    alpha_rotation = 0

    image, image_label = generate_contour_image(
        frag=fragment,
        frag_params=gabor_parameters,
        c_len=contour_len,
        beta=beta_rotation,
        alpha=alpha_rotation,
        f_tile_size=full_tile_size,
        img_size=image_size,
        random_alpha_rot=True
    )
    print(image_label)

    plt.figure()
    plt.imshow(image)
    plt.title("Input Image")

    # Highlight the label
    center_full_tile_start = image_size[:2] // 2 - (full_tile_size[0:2] // 2)
    full_tile_starts = get_background_tiles_locations(
        frag_len=full_tile_size[0],
        img_len=image_size[0],
        row_offset=0,
        space_bw_tiles=0,
        tgt_n_visual_rf_start=center_full_tile_start[0]
    )

    contour_containing_tiles = full_tile_starts[image_label.flatten().nonzero()]
    labeled_image = highlight_tiles(image, full_tile_size, contour_containing_tiles)

    plt.figure()
    plt.imshow(labeled_image)
    plt.title("Labeled image")

    input("press any key to exit")
