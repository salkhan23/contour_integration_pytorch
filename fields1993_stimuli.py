# ---------------------------------------------------------------------------------------
# Generate Contour Stimuli embedded in a sea of distractions similar to
# Field, Hayes & Hess - 1993 - Contour Integration by the Human Visual System: Evidence
# for a local association field "
#
# NOTE: This is deep learning library independent
# ---------------------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
import gabor_fits
import copy
from skimage.transform import rotate
import os
from datetime import datetime
import sys
import shutil


D_JITTER_SCALE = 8   # Relative to distance (d) between contour fragment.
# If used adds a random +- d // D_JITTER_SCALE to the distance between contour fragments.


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
    return rotate(tile, angle=(np.random.randint(0, int(num_possible_rotations)) * delta_rotation))


def tile_image(img, frag, insert_loc_arr, rotate_frags=True, delta_rotation=45, gaussian_smoothing=True,
               sigma=4.0, replace=True):
    """
    Place tile 'fragments' at the specified starting positions (x, y) in the image.

    :param replace: If True, will replace image pixels with the tile. If False will multiply
           image pixels and tile values. Default = True
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

            start_x_loc = int(max(x_arr[idx], 0))
            stop_x_loc = int(min(x_arr[idx] + tile_len, img.shape[0]))

            start_y_loc = int(max(y_arr[idx], 0))
            stop_y_loc = int(min(y_arr[idx] + tile_len, img.shape[1]))

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
            if replace:
                new_img_pixels = \
                    tile[tile_x_start: tile_x_start + stop_x_loc - start_x_loc,
                         tile_y_start: tile_y_start + stop_y_loc - start_y_loc, :]
            else:
                new_img_pixels = \
                    img[start_x_loc: stop_x_loc, start_y_loc: stop_y_loc, :] * \
                    tile[tile_x_start: tile_x_start + stop_x_loc - start_x_loc,
                         tile_y_start: tile_y_start + stop_y_loc - start_y_loc, :]

            img[start_x_loc: stop_x_loc, start_y_loc: stop_y_loc, :] = new_img_pixels

    return img


def _add_single_side_of_contour_constant_separation(
        img, center_frag_start, frag, frag_params, c_len, beta, alpha, d, d_delta, frag_size,
        random_beta_rot=False, random_alpha_rot=True, use_d_jitter=True):
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

    tile_offset = np.zeros((2,), dtype=int)
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

        # In figure 5, (but not in the text),a random jitter of [+- d/4] is added to the distance between
        # fragments, this presumably to prevent relying on eqi-distance between fragments.
        if use_d_jitter:
            d_jitter = d // D_JITTER_SCALE * np.random.uniform(-1, 1)
        else:
            d_jitter = 0
        # print("d_jitter {}".format(d_jitter))

        # Note
        # [1] Origin of (x, y) top left corner
        # [2] Dim 0 increases downward direction, Dim 1 increases in the right direction
        # [3] Gabor angles are specified wrt y-axis i.e. 0 orientation is vertical. For position
        #     we need the angles to be relative to the x-axis.
        tile_offset[0] = (d + d_jitter) * np.cos(loc_angle / 180.0 * np.pi)
        tile_offset[1] = (d + d_jitter) * np.sin(loc_angle / 180.0 * np.pi)

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
            # print("Tile {0} overlaps with tile at location {1}".format(curr_tile_start, prev_tile_start))
            tile_offset[0] += d_delta * np.cos(loc_angle / 180.0 * np.pi)
            tile_offset[1] += d_delta * np.sin(loc_angle / 180.0 * np.pi)

            curr_tile_start = prev_tile_start + tile_offset
            # print("Current tile relocated to {0}. (offsets {1})".format(curr_tile_start, tile_offset))

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

    return img, tile_starts, acc_angle


def add_contour_path_constant_separation(
        img, frag, frag_params, c_len, beta, alpha, d, center_frag_start=None, use_d_jitter=True,
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
    :param use_d_jitter: if Set, a random d // D_JITTER_SCALE is added to the distance between contour fragments
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

    img, tiles, final_acc_angle_end = _add_single_side_of_contour_constant_separation(
        img, center_frag_start, frag, frag_params, c_len, beta, alpha, d, d_delta, frag_size,
        random_beta_rot=rand_inter_frag_direction_change,
        random_alpha_rot=random_alpha_rot,
        use_d_jitter=use_d_jitter
    )

    c_tile_starts.extend(tiles)

    if base_contour == 'circle':
        beta = -beta
    elif base_contour == 'random':
        beta = np.random.choice((-1, 1), size=1) * beta

    img, tiles, final_acc_angle_start = _add_single_side_of_contour_constant_separation(
        img, center_frag_start, frag, frag_params, c_len, beta, alpha, -d, -d_delta, frag_size,
        random_beta_rot=rand_inter_frag_direction_change,
        random_alpha_rot=random_alpha_rot,
        use_d_jitter=use_d_jitter
    )

    # Attach in reverse order  to the top of the list
    tiles.reverse()
    c_tile_starts = tiles + c_tile_starts

    # ---------------------------
    c_tile_starts = np.array(c_tile_starts)

    return img, c_tile_starts, final_acc_angle_end, final_acc_angle_start


def get_non_overlapping_bg_fragment(
        f_tile_start, f_tile_size, c_tile_starts, c_tile_size, max_offset, no_overlap_th=0.4):
    """

    :param no_overlap_th:
    :param f_tile_size:
    :param f_tile_start:
    :param c_tile_starts:
    :param c_tile_size:
    :param max_offset:
    :return:
    """
    # print("get_non_overlapping_bg_fragment: full tile start: {}".format(f_tile_start))

    no_overlapping_starts = []

    for r_idx in range(max_offset):
        for c_idx in range(max_offset):

            l1 = f_tile_start + np.array([r_idx, c_idx])  # top left corner of new bg tile
            r1 = l1 + c_tile_size   # lower right corner of new bg tile

            is_overlapping = False

            # print("Checking start location (y,x) = {}".format(l1))

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
                no_overlapping_starts.append(l1)

    num_no_overlap = len(no_overlapping_starts)
    total_location_checked = max_offset ** 2
    no_overlap_percent = num_no_overlap / total_location_checked

    start_loc = None
    if no_overlap_percent > no_overlap_th:
        start_loc_idx = np.random.choice(len(no_overlapping_starts))
        start_loc = no_overlapping_starts[start_loc_idx]

    # print("Non-overlapping start_loc {}. Locations checked: Non_overlap/Total = {}/{} = {:0.2f}".format(
    #     start_loc, total_location_checked, num_no_overlap, no_overlap_percent))

    return start_loc


def get_background_tiles_locations(frag_len, img_len, row_offset, space_bw_tiles, tgt_n_visual_rf_start=None):
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
    frag_spacing = int(frag_len + space_bw_tiles)
    n_tiles = int(img_len // frag_spacing)

    # To handle non-zero row shift, we have to add additional tiles, so that the whole image is populated
    add_tiles = 0
    if row_offset:
        max_shift = (n_tiles // 2 + 1) * row_offset
        add_tiles = abs(max_shift) // frag_spacing + 1
    # print("Number of tiles in image %d, number of additional tiles %d" % (n_tiles, add_tiles))

    n_tiles += add_tiles
    if n_tiles & 1 == 1:  # make even
        n_tiles += 1

    if tgt_n_visual_rf_start is None:
        tgt_n_visual_rf_start = img_len // 2 - frag_len // 2  # Tile @ center of image

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
        row_offset = int(frag_spacing / float(frag_len) * row_offset)

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


def find_containing_tile_idx(p, tile_start_arr, tile_size):
    """
    Find the index of tile that contains point p. Tiles are specified using the the
    array tile_start_arr (top left corner) and of size tile_size
    """
    x_greater_start = p[0] >= tile_start_arr[:, 0]
    x_less_stop = p[0] < (tile_start_arr[:, 0] + tile_size[0])
    y_greater_start = p[1] >= tile_start_arr[:, 1]
    y_less_stop = p[1] < (tile_start_arr[:, 1] + tile_size[1])

    tmp = x_greater_start * x_less_stop * y_greater_start * y_less_stop

    return int(np.nonzero(tmp)[0])


def add_background_fragments(img, frag, c_frag_starts, f_tile_size, delta_rotation, frag_params,
                             relocate_allowed=True):
    """
    Divides the image into f(ull)_tile_size, for all locations that do not contain a contour
    fragment add a background fragment at a random location with a random orientation.

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
    remove_bg_frag_idxs = []
    check_for_overlap_idxs = []
    removed_bg_frag_starts = []

    for c_frag_start in c_frag_starts:
        # Which full tiles do all four corners of the contour fragment lie in
        tl_idx = find_containing_tile_idx(c_frag_start, f_tile_starts, f_tile_size)
        tr_idx = find_containing_tile_idx(
            c_frag_start + np.array([frag.shape[0], 0]), f_tile_starts, f_tile_size)
        bl_idx = find_containing_tile_idx(
            c_frag_start + np.array([0, frag.shape[0]]), f_tile_starts, f_tile_size)
        br_idx = find_containing_tile_idx(
            c_frag_start + frag.shape[0:2], f_tile_starts, f_tile_size)

        if tl_idx == tr_idx == bl_idx == br_idx:
            # if all in same full tile, remove corresponding bg tile
            remove_bg_frag_idxs.append(tl_idx)
            removed_bg_frag_starts.append(bg_frag_starts[tl_idx, :])
        else:
            # if contour fragment lies in multiple full tiles check for possible
            # overlap with bg tile fragment.
            check_for_overlap_idxs.extend([tl_idx, tr_idx, bl_idx, br_idx])

    # Remove duplicates
    remove_bg_frag_idxs = list(set(remove_bg_frag_idxs))
    check_for_overlap_idxs = list(set(check_for_overlap_idxs))
    # Also remove any to check_for_overlapping idxs that are also removed
    check_for_overlap_idxs = \
        [idx for idx in check_for_overlap_idxs if idx not in remove_bg_frag_idxs]

    relocate_bg_frag_starts = []
    for idx in check_for_overlap_idxs:
        # print("Check for overlap in full tile {}. Bg frag tile location {} @ index {}".format(
        #     f_tile_starts[idx, :], bg_frag_starts[idx, ], idx ))

        tgt_bg_frag = np.expand_dims(bg_frag_starts[idx, ], axis=0)
        f_tile_start = f_tile_starts[idx, :]

        dist_to_c_frag = np.linalg.norm(tgt_bg_frag - c_frag_starts, axis=1)
        # for ii, dist in enumerate(dist_to_c_frag):
        #     print("{0}: {1}".format(ii, dist))

        if any(dist_to_c_frag <= np.sqrt(2)*frag.shape[0]):
            # print("bg fragment overlaps with contour fragment")
            no_overlap_bg_frag = None
            if relocate_allowed:
                no_overlap_bg_frag = get_non_overlapping_bg_fragment(
                    f_tile_start=f_tile_start,
                    f_tile_size=f_tile_size,
                    c_tile_starts=c_frag_starts,
                    c_tile_size=frag.shape[0:2],
                    max_offset=max_displace,
                )

            if no_overlap_bg_frag is None:
                # print("\tRemove bg tile")
                # Cannot Delete when indexing over the list
                # bg_frag_starts = np.delete(bg_frag_starts, idx, axis=0)
                removed_bg_frag_starts.append(bg_frag_starts[idx, :])
                remove_bg_frag_idxs.append(idx)

            else:
                # print("\tRelocate bg tile to {}".format(no_overlap_bg_frag))
                bg_frag_starts[idx, :] = no_overlap_bg_frag
                relocate_bg_frag_starts.append(no_overlap_bg_frag)

    # Remove the cannot be replaced bg fragments
    bg_frag_starts = np.delete(bg_frag_starts, remove_bg_frag_idxs, axis=0)

    # Now add the background fragment tiles
    # -------------------------------------
    if type(frag_params) is not list:
        frag_params = [frag_params]
    rotated_frag_params_list = copy.deepcopy(frag_params)

    num_possible_rotations = 360 // delta_rotation

    for start in bg_frag_starts:

        random_rotation = np.random.randint(0, int(num_possible_rotations)) * delta_rotation
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

    removed_bg_frag_starts = np.array(removed_bg_frag_starts)
    relocate_bg_frag_starts = np.array(relocate_bg_frag_starts)

    return img, bg_frag_starts, removed_bg_frag_starts, relocate_bg_frag_starts


def highlight_tiles(
        in_img, tile_shape, insert_loc_arr, edge_color=(255, 0, 0), edge_width=1,
        force_color=False):
    """
    Highlight specified tiles in the image


    :param force_color:
    :param edge_width:
    :param in_img:
    :param tile_shape:
    :param insert_loc_arr:
    :param edge_color:

    :return: output image with the tiles highlighted
    """
    out_img = np.copy(in_img).astype('uint16')  # This to prevent wrap around of values

    img_size = in_img.shape[:2]
    tile_len = tile_shape[0]

    if in_img.mean() > 200:
        # white background, force boundary colours to what is specified instead of adding them on.
        # label will be visible at least but combining multiple label son the image wont work.
        force_color = True

    if insert_loc_arr.ndim == 1:
        x_arr = [insert_loc_arr[0]]
        y_arr = [insert_loc_arr[1]]
    else:
        x_arr = insert_loc_arr[:, 0]
        y_arr = insert_loc_arr[:, 1]

    edge_color = np.array(edge_color).astype('uint16')

    for idx in range(len(x_arr)):
        # print("idx {}, x={},y={}".format(idx, x_arr[idx], y_arr[idx]))

        if (-tile_len < x_arr[idx] < img_size[0]) and (-tile_len < y_arr[idx] < img_size[1]):

            start_x_loc = max(x_arr[idx], 0)
            stop_x_loc = min(x_arr[idx] + tile_len, img_size[0]-1)

            start_y_loc = max(y_arr[idx], 0)
            stop_y_loc = min(y_arr[idx] + tile_len, img_size[1]-1)

            # print("Highlight tile @ tl=({0}, {1}), br=({2},{3})".format(
            #     start_x_loc, start_y_loc, stop_x_loc, stop_y_loc))

            if force_color:
                out_img[
                    start_x_loc: stop_x_loc + 1,
                    start_y_loc: start_y_loc + edge_width,
                    :
                ] = edge_color

                out_img[
                    start_x_loc: stop_x_loc + 1,
                    stop_y_loc: stop_y_loc + edge_width,
                    :
                ] = edge_color

                out_img[
                    start_x_loc: start_x_loc + edge_width,
                    start_y_loc: stop_y_loc + 1,
                    :
                ] = edge_color

                out_img[
                    stop_x_loc: stop_x_loc + edge_width,
                    start_y_loc: stop_y_loc + edge_width,
                    :
                ] = edge_color

            else:
                out_img[
                    start_x_loc: stop_x_loc + 1,
                    start_y_loc: start_y_loc + edge_width,
                    :
                ] += edge_color

                out_img[
                    start_x_loc: stop_x_loc + 1,
                    stop_y_loc: stop_y_loc + edge_width,
                    :
                ] += edge_color

                out_img[
                    start_x_loc: start_x_loc + edge_width,
                    start_y_loc: stop_y_loc + 1,
                    :
                ] += edge_color

                out_img[
                    stop_x_loc: stop_x_loc + edge_width,
                    start_y_loc: stop_y_loc + edge_width,
                    :
                ] += edge_color

                out_img[out_img > 255] = 255

    out_img = out_img.astype('uint8')

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
        use_d_jitter=True, rand_inter_frag_direction_change=True, random_alpha_rot=True,
        center_frag_start=None, base_contour='random', bg=None):
    """

    Generate image and label for specified fragment parameters.

    In [Fields -1993] a small visible stimulus is placed inside a large tile
    Here, full tile refers to the large tile & fragment tile refers to the visible stimulus. If the
    fragment is part of the contour, it is called a contour fragment otherwise a background fragment

    :param bg: value to use for bg. If None, will be calculated using boundary pixels of fragment
    :param frag:
    :param frag_params:
    :param c_len:
    :param beta:
    :param alpha:
    :param f_tile_size:
    :param img_size: [Default = (227, 227, 3)]
    :param bg_frag_relocate: if a bg fragment overlaps with a contour fragment within a tile.
           Try to relocate it before removing it.
    :param rand_inter_frag_direction_change: Curve Direction changes randomly between component
            fragments. [Default =True]
    :param random_alpha_rot:
    :param center_frag_start: starting location of contour fragments
    :param use_d_jitter: if set a random d / D_JITTER_SCALE is added to distance between fragments
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

    if bg is None:
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
    f_tiles_single_dim = int(np.sqrt(num_f_tiles))

    img = np.ones(img_size, dtype=np.uint8) * bg

    label = np.zeros(num_f_tiles, dtype='uint8')

    # Image  -------------------------------------
    # Get Contour Fragments
    if (c_len > 1) or (c_len == 1 and beta == 0):
        img, c_frag_starts, final_acc_angle_end, final_acc_angle_start = \
            add_contour_path_constant_separation(
                img, frag, frag_params, c_len, beta, alpha, f_tile_size[0],
                center_frag_start=center_frag_start,
                rand_inter_frag_direction_change=rand_inter_frag_direction_change,
                random_alpha_rot=random_alpha_rot,
                base_contour=base_contour,
                use_d_jitter=use_d_jitter,
            )
    else:
        c_frag_starts = []
        final_acc_angle_start = 0
        final_acc_angle_end = 0
        # For all other cases, c_len ==1 and beta !=0, just add background tiles.
        # which move around within the full tile.
        # No contour integration for single contour fragments

    # print("contour fragment starts:\n{}".format(c_frag_starts))

    # # Add background fragments
    img, bg_frag_starts, removed_tiles, relocated_tiles = add_background_fragments(
        img, frag, c_frag_starts, f_tile_size, 1, frag_params, bg_frag_relocate)

    # Label -----------------------------------
    if c_len > 1:  # No contour integration for single fragments
        for c_frag_start in c_frag_starts:
            c_frag_center = c_frag_start + frag_size // 2

            dist_to_c_frag = np.linalg.norm(f_tile_centers - c_frag_center, axis=1)
            closest_full_tile_idx = np.argmin(dist_to_c_frag)

            # print("Closest Full Tile Index {}. Distance {}".format(
            #     closest_full_tile_idx, dist_to_c_frag[closest_full_tile_idx]))

            if dist_to_c_frag[closest_full_tile_idx] <= np.sqrt(2) * (f_tile_size[0] / 2.):
                # print("Added")
                label[closest_full_tile_idx] = 1

    label = label.reshape(f_tiles_single_dim, f_tiles_single_dim)

    # # Debug ------------------------
    # plt.figure()
    # plt.imshow(img)
    # plt.title("Original Image")
    #
    # # Highlight Contour tiles - Red
    # img = highlight_tiles(img, frag_size, c_frag_starts,  edge_color=(255, 0, 0))
    #
    # # Highlight Background Fragment tiles - Green
    # img = highlight_tiles(img, frag_size, bg_frag_starts, edge_color=(0, 255, 0))
    #
    # # Highlight Removed tiles - Blue
    # if len(removed_tiles) != 0:
    #     img = highlight_tiles(img, frag_size, removed_tiles, edge_color=(0, 0, 255))
    #
    # # Highlight Relocated tiles - Teal
    # if len(relocated_tiles) != 0:
    #     img = highlight_tiles(img, frag_size, relocated_tiles, edge_color=(0, 255, 255))
    #
    # # highlight full tiles - Yellow
    # img = highlight_tiles(img, f_tile_size, f_tile_starts, edge_color=(255, 255, 0))
    #
    # plt.figure()
    # plt.imshow(img)
    # print(label)
    # import pdb
    # pdb.set_trace()

    return img, label, c_frag_starts, final_acc_angle_end, final_acc_angle_start


def get_contour_start_ranges(c_len, frag_orient, f_tile_size, img_size, beta=15):
    """
    Get the min, max starting offsets so that the defined contours stays within the image

    :param c_len:
    :param frag_orient:
    :param f_tile_size:
    :param img_size:
    :param beta:

    :return: two tuples: (x_min, x_max), (y_min, y_max)
    """
    # The additional f_tile_size[0] // 4 accounts for the distance jitter added to each fragment
    max_inter_frag_dist = (f_tile_size[0] + f_tile_size[0] // D_JITTER_SCALE)

    half_cont_h = (c_len * max_inter_frag_dist * np.cos(frag_orient / 180. * np.pi)) // 2
    half_cont_w = (c_len * max_inter_frag_dist * np.sin(frag_orient / 180. * np.pi)) // 2

    # Get max displacement if contour is curving
    acc_angle = frag_orient
    acc_h_rot = 0
    acc_w_rot = 0
    h_rot_list = [0]
    w_rot_list = [0]

    for c_idx in range(c_len // 2):
        acc_angle += beta

        acc_h_rot += max_inter_frag_dist * np.cos(acc_angle * np.pi / 180.0)
        acc_w_rot += max_inter_frag_dist * np.sin(acc_angle * np.pi / 180.0)

        h_rot_list.append(acc_h_rot)
        w_rot_list.append(acc_w_rot)

    max_h_rot = max(np.abs(h_rot_list))
    max_w_rot = max(np.abs(w_rot_list))

    # Max any displacement
    max_h_displace = max(abs(half_cont_h), abs(max_h_rot))
    max_w_displace = max(abs(half_cont_w), abs(max_w_rot))

    # print("Contour max height {}, max rot displace {}. Choose {}".format(half_cont_h, max_h_rot, max_h_displace))
    # print("Contour max width {}, max rot displace {}. Choose {}".format(half_cont_w, max_w_rot, max_w_displace))

    # Now calculate min/max starting positions
    max_h_offset = np.max((img_size[0] // 2) - max_h_displace, 0)
    max_w_offset = np.max((img_size[0] // 2) - max_w_displace, 0)

    min_start_h = max((img_size[0] // 2) - max_h_offset, f_tile_size[0] // 2)
    max_start_h = min((img_size[0] // 2) + max_h_offset, img_size[0] - f_tile_size[0] // 2)

    min_start_w = max((img_size[1] // 2) - max_w_offset, f_tile_size[1] // 2)
    max_start_w = min((img_size[1] // 2) + max_w_offset, img_size[1] - f_tile_size[1] // 2)

    # print("h start range [{}, {}]. w start ranges [{}, {}]".format(
    #     min_start_h, max_start_h, min_start_w, max_start_w))

    return (min_start_h, max_start_h), (min_start_w, max_start_w)


def generate_data_set(
        n_imgs_per_set, base_dir, frag_tile_size, frag_params_list, c_len_arr, beta_rot_arr, alpha_rot_arr, f_tile_size,
        img_size=None, use_d_jitter=True, rand_inter_frag_direction_change=True, random_alpha_rot=False,
        center_frag_start=None, bg_frag_relocate=True):
    """
     Generate Data Set
     TODO: handle the case when multiple gabor fragments / tile sizes are defined.
     TODO: handle the getting orientation for a Gabor with Three channels.

    :param bg_frag_relocate:
    :param center_frag_start:
    :param rand_inter_frag_direction_change:
    :param use_d_jitter:
    :param random_alpha_rot:
    :param n_imgs_per_set:
    :param base_dir:
    :param frag_tile_size:
    :param frag_params_list: list of list of frag parameters. Each list of frag params can contain
    one or 3 gabor parameters dictionary of each channel. Each dictionary should have the following parameters:
        {
            'x0': 0,
            'y0': 0,
            'theta_deg': 0,
            'amp': 1,
            'sigma': 4.0,
            'lambda1': 10,
            'psi': 0,
            'gamma': 1
            'bg' = optional, bg value to set for generated images.
        }
    :param c_len_arr:
    :param beta_rot_arr:
    :param alpha_rot_arr:
    :param f_tile_size:
    :param img_size:
    :return:
    """

    if os.path.exists(base_dir):
        ans = input("Data already exists in {}.\nEnter y to overwrite".format(base_dir))
        if 'y' in ans.lower():
            shutil.rmtree(base_dir)
        else:
            sys.exit()

    # Stores the path to all files in the image_set common to images and labels
    data_key = 'data_key.txt'

    data_store_dir = os.path.join(base_dir, 'images')
    labels_store_dir = os.path.join(base_dir, 'labels')
    data_key_file = os.path.join(base_dir, data_key)

    start_time = datetime.now()
    n_total_imgs = 0
    n_invalid_imgs = 0  # this is only for debug. Invalid images are removed

    if not os.path.exists(data_store_dir):
        os.makedirs(data_store_dir)
    if not os.path.exists(labels_store_dir):
        os.makedirs(labels_store_dir)

    f = open(data_key_file, 'w+')

    for frag_param_idx, frag_params in enumerate(frag_params_list):
        print("{0} Parameter Set {1} {0}".format('*'*30, frag_param_idx))

        # make the frag params folder
        frag_param_dir = 'frag_{}'.format(frag_param_idx)

        frag = gabor_fits.get_gabor_fragment(frag_params, frag_tile_size)

        # Get background pixel value if it exists value if exists
        if 'bg' in frag_params[0].keys():
            bg = frag_params[0]['bg']
        else:
            bg = None

        for c_len in c_len_arr:
            c_len_name = os.path.join(frag_param_dir, 'clen_{}'.format(c_len))

            if center_frag_start is None:
                x_start_range, y_start_range = get_contour_start_ranges(
                    c_len=c_len,
                    frag_orient=frag_params[0]['theta_deg'],  # Todo handle the case when there are three orientations
                    f_tile_size=f_tile_size,
                    img_size=img_size
                )
            else:
                x_start_range = y_start_range = 0

            for beta in beta_rot_arr:
                c_len_beta_rot_dir = os.path.join(c_len_name, 'beta_{}'.format(beta))

                for alpha in alpha_rot_arr:
                    c_len_beta_rot_alpha_rot_dir = os.path.join(c_len_beta_rot_dir, 'alpha_{}'.format(alpha))

                    store_data_dir_full = os.path.join(data_store_dir, c_len_beta_rot_alpha_rot_dir)
                    store_label_dir_full = os.path.join(labels_store_dir, c_len_beta_rot_alpha_rot_dir)

                    if not os.path.exists(store_data_dir_full):
                        os.makedirs(store_data_dir_full)
                    if not os.path.exists(store_label_dir_full):
                        os.makedirs(store_label_dir_full)

                    print("Param Set {}. Generating {} images with c_len = {}, beta = {}, alpha = {}, bg={}".format(
                        frag_param_idx, n_imgs_per_set, c_len, beta, alpha, bg))

                    n_imgs_current_set = 0
                    for i_idx in range(n_imgs_per_set + 50):
                        if center_frag_start is None:
                            middle_frag_start = np.array([
                                np.random.randint(x_start_range[0], x_start_range[1]),
                                np.random.randint(y_start_range[0], y_start_range[1]),
                            ])
                        else:
                            middle_frag_start = center_frag_start

                        # print("print middle_frag_start {}".format(middle_frag_start))
                        img, img_label, _, _, _ = generate_contour_image(
                            frag=frag,
                            frag_params=frag_params,
                            c_len=c_len,
                            beta=beta,
                            alpha=alpha,
                            f_tile_size=f_tile_size,
                            img_size=img_size,
                            random_alpha_rot=random_alpha_rot,
                            center_frag_start=middle_frag_start,
                            base_contour='random',
                            use_d_jitter=use_d_jitter,
                            rand_inter_frag_direction_change=rand_inter_frag_direction_change,
                            bg_frag_relocate=bg_frag_relocate,
                            bg=bg,
                        )

                        if is_label_valid(img_label):
                            # Save
                            file_name = 'clen_{}_beta_{}_alpha_{}_{}'.format(c_len, beta, alpha, i_idx)
                            plt.imsave(
                                fname=os.path.join(store_data_dir_full, file_name + '.png'), arr=img, format='PNG')
                            np.save(file=os.path.join(store_label_dir_full, file_name + '.npy'), arr=img_label)
                            f.write(os.path.join(c_len_beta_rot_alpha_rot_dir, file_name) + '\n')

                            n_total_imgs += 1

                            n_imgs_current_set += 1
                            if n_imgs_current_set >= n_imgs_per_set:
                                break

                        else:
                            n_invalid_imgs += 1
                            print("Number of invalid Images {}.".format(n_invalid_imgs))

                            # plot_label_on_image(img, img_label, f_tile_size, edge_color=(250, 0, 0), edge_width=3)
                            # import pdb
                            # pdb.set_trace()

    f.close()
    print("Data set created @ {}.\nContains {} Images. Time Taken {}".format(
          base_dir, n_total_imgs, datetime.now() - start_time))


def plot_label_on_image(
        img, label, f_tile_size, edge_color=(255, 0, 0), edge_width=1, display_figure=True):
    f_tile_starts = get_background_tiles_locations(
        frag_len=f_tile_size[0],
        img_len=np.max(img.shape),
        row_offset=0,
        space_bw_tiles=0,
    )

    contour_containing_tiles = f_tile_starts[label.flatten().nonzero()]
    labeled_image = highlight_tiles(
        img, f_tile_size, contour_containing_tiles, edge_color, edge_width)

    if display_figure:
        plt.figure()
        plt.imshow(labeled_image)
        plt.title("Labeled image")

    return labeled_image


def is_label_valid(label, n_contours=1):
    """
    Looks for any discontinuities between labels.

    :param n_contours:
    :param label:
    :return:
    """
    is_valid = False

    ones_idxs = np.argwhere(label >= 1)
    ones_row = ones_idxs[:, 0]
    ones_col = ones_idxs[:, 1]

    num_ends = 0
    for tgt_idx in range(len(ones_idxs)):
        row_idxs = np.arange(ones_row[tgt_idx] - 1, ones_row[tgt_idx] + 2)
        col_idxs = np.arange(ones_col[tgt_idx] - 1,  ones_col[tgt_idx] + 2)

        neigbors = label[row_idxs[0]:row_idxs[-1] + 1, col_idxs[0]:col_idxs[-1] + 1]
        if neigbors.sum() <= 2:
            num_ends += 1

        # print("tgt row {}. Indices {}".format(ones_row[tgt_idx], row_idxs))
        # print("tgt col {}. Indices {}".format(ones_col[tgt_idx], col_idxs))
        # print("neigbors {}".format(neigbors))
        # print("Neighbors sum {}".format(neigbors.sum()))

    if num_ends <= 2 * n_contours:
        is_valid = True
    # else:
    #     print("Contour has discontinuities. Contours with only one neighbor {}. "
    #           "Expected {}".format(num_ends, 2*n_contours))

    return is_valid


if __name__ == "__main__":
    # -----------------------------------------------------------------------------------
    # Initialization
    # -----------------------------------------------------------------------------------
    random_seed = 10

    fragment_size = (7, 7)
    full_tile_size = np.array([14, 14])

    image_size = np.array([256, 256, 3])

    # Immutable
    plt.ion()
    np.random.seed(random_seed)

    # -----------------------------------------------------------------------------------
    # Gabor Fragment
    # -----------------------------------------------------------------------------------
    # BW Gabor, Single Channel (white on black Background)
    # ----------------------------------------------------
    bg_value = 0
    gabor_parameters_list = [{
        'x0': 0,
        'y0': 0,
        'theta_deg': 0,
        'amp': 1,
        'sigma': 4.0,
        'lambda1': 7,
        'psi': 0,
        'gamma': 1
    }]

    # # BW Gabor, Single Channel (white on black Background)
    # # ----------------------------------------------------
    # bg_value = 255
    # gabor_parameters_list = [{
    #     'x0': 0,
    #     'y0': 0,
    #     'theta_deg': -45,
    #     'amp': -0.46,
    #     'sigma': 0.9,
    #     'lambda1': 20,
    #     'psi': 0,
    #     'gamma': 0
    # }]

    # # Colored Gabor, 3 Channels
    # # -------------------------
    # bg_value = None
    # gabor_parameters_list = [
    #     {
    #         'x0': 0.76,
    #         'y0': -0.40,
    #         'theta_deg': 38.23,
    #         'amp': 0.53,
    #         'sigma': 4.0,
    #         'lambda1': 10.68,
    #         'psi': -0.91,
    #         'gamma': 1.22
    #     },
    #     {
    #         'x0': 0.28,
    #         'y0': 0.28,
    #         'theta_deg': 37.22,
    #         'amp': 0.28,
    #         'sigma': 4.0,
    #         'lambda1': 13.44,
    #         'psi': 1.80,
    #         'gamma': 1.18
    #     },
    #     {
    #         'x0': 1.01,
    #         'y0': -0.75,
    #         'theta_deg': 38.66,
    #         'amp': 0.30,
    #         'sigma': 4.0,
    #         'lambda1': 9.27,
    #         'psi': 2.54,
    #         'gamma': 1.32
    #     }
    # ]

    fragment = gabor_fits.get_gabor_fragment(gabor_parameters_list, fragment_size)
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
    # test_image, path_fragment_starts, _, _ = add_contour_path_constant_separation(
    #     img=test_image,
    #     frag=fragment,
    #     frag_params=gabor_parameters_list,
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
    #     gabor_parameters_list,
    #     relocate_allowed=True
    # )
    # plt.figure()
    # plt.imshow(test_image)
    # plt.title("Highlighted Tiles: Stimulus c_len = {}, beta = {}, alpha = {}".format(
    #     contour_len, beta_rotation, alpha_rotation))
    # # Highlight Tiles
    # full_tile_starts = get_background_tiles_locations(
    #     frag_len=full_tile_size[0],
    #     img_len=image_size[1],
    #     row_offset=0,
    #     space_bw_tiles=0,
    #     tgt_n_visual_rf_start=image_size[0] // 2 - (full_tile_size[0] // 2)
    # )
    #
    # test_image = highlight_tiles(
    #     test_image, fragment_size, bg_tiles, edge_color=(255, 255, 0))
    # test_image = highlight_tiles(
    #     test_image, full_tile_size, full_tile_starts, edge_color=(255, 0, 0))
    # test_image = highlight_tiles(
    #     test_image, fragment_size, path_fragment_starts, edge_color=(0, 0, 255))
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

    image, image_label, _, _, _ = generate_contour_image(
        frag=fragment,
        frag_params=gabor_parameters_list,
        c_len=contour_len,
        beta=beta_rotation,
        alpha=alpha_rotation,
        f_tile_size=full_tile_size,
        img_size=image_size,
        random_alpha_rot=True,
        rand_inter_frag_direction_change=False,
        use_d_jitter=True,
        bg=bg_value,
        bg_frag_relocate=True
    )
    # print(image_label)
    # print("Label is valid? {}".format(is_label_valid(image_label)))

    plt.figure()
    plt.imshow(image)
    plt.title("Input Image")

    plot_label_on_image(
        image, image_label, full_tile_size, edge_color=(250, 0, 0), edge_width=1)

    input("press any key to exit")

    # # -----------------------------------------------------------------------------------
    # # Gabor Fragments from List
    # # -----------------------------------------------------------------------------------
    # import pickle
    #
    # gabor_params_file = 'channel_wise_optimal_stimuli.pickle'
    # with open(gabor_params_file, 'rb') as handle:
    #     data = pickle.load(handle)
    #
    # base_gabors_params_list = None
    # if type(data) is list:
    #     gabor_parameters_list = data
    # elif type(data) is dict:
    #     gabor_parameters_list = data['list_of_optimal_stimuli']
    #     base_gabors_params_list = data['base_gabor_params']
    # else:
    #     raise Exception("Unknown pickle file type")
    #
    # for idx in range(64):
    #     contour_len = 1
    #     beta_rotation = 15
    #     alpha_rotation = 0
    #
    #     fragment = gabor_fits.get_gabor_fragment(gabor_parameters_list[idx], fragment_size)
    #
    #     x_start_range, y_start_range = get_contour_start_ranges(
    #         c_len=contour_len,
    #         frag_orient=gabor_parameters_list[idx][0]['theta_deg'],
    #         # Todo handle the case when there are three orientations
    #         f_tile_size=full_tile_size,
    #         img_size=image_size
    #     )
    #
    #     middle_frag_start = np.array([
    #         np.random.randint(x_start_range[0], x_start_range[1]),
    #         np.random.randint(y_start_range[0], y_start_range[1]),
    #     ])
    #
    #     image, image_label, _, _, _ = generate_contour_image(
    #         frag=fragment,
    #         frag_params=gabor_parameters_list[idx],
    #         c_len=contour_len,
    #         beta=beta_rotation,
    #         alpha=alpha_rotation,
    #         f_tile_size=full_tile_size,
    #         img_size=image_size,
    #         random_alpha_rot=True,
    #         rand_inter_frag_direction_change=True,
    #         use_d_jitter=True,
    #         bg=gabor_parameters_list[idx][0]['bg'],
    #         center_frag_start=middle_frag_start,
    #     )
    #     print(image_label)
    #     print("Label is valid? {}".format(is_label_valid(image_label)))
    #
    #     plt.figure()
    #     plt.imshow(image)
    #     plt.title("Input Image")
    #
    #     plot_label_on_image(
    #         image, image_label, full_tile_size, edge_color=(250, 0, 0), edge_width=1)
    #
    #     import pdb
    #     pdb.set_trace()
    #     plt.close('all')

    # -----------------------------------------------------------------------------------
    # End
    # -----------------------------------------------------------------------------------
    print("End")
    import pdb
    pdb.set_trace()
