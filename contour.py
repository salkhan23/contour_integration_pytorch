# This code picks a contour at random.

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as img
import pdb


def get_random_point_on_an_edge(image):
    """
    :param image: 2D array of edge (1) and non-edge (0) pixels
    :return: row,col of a random point that is on one of the edges
    """
    edge_indices = np.where(image.flatten())[0]
    random_edge_index = np.random.choice(edge_indices)
    row_ind = int(np.floor(random_edge_index / image.shape[1]))
    col_ind = random_edge_index - row_ind*image.shape[1]
    return (row_ind, col_ind)


def get_neighbourhood(image, point):
    """
    :param image: 2D array of edge (1) and non-edge (0) pixels
    :param point: row, col of a point in the image
    :return: 3x3 patch of image with point at centre
    """
    def get_offset_range(image_dimension, coordinate):
        if coordinate == 0:
            return [0,1]
        elif coordinate == image_dimension - 1:
            return [-1,0]
        else:
            return [-1,1]

    ro = get_offset_range(image.shape[0], point[0])
    co = get_offset_range(image.shape[1], point[1])
    p = point

    result = np.zeros((3,3))
    neighbourhood = image[p[0]+ro[0]:p[0]+ro[1]+1, p[1]+co[0]:p[1]+co[1]+1]
    result[1+ro[0]:1+ro[1]+1, 1+co[0]:1+co[1]+1] = neighbourhood
    return result


def has_clean_line(neighbourhood):
    """
    :param neighbourhood: 3x3 patch of edge image
    :return: true of the patch contains exactly three edge points which are
        in a straight line
    """
    result = False
    if np.sum(neighbourhood) == 3:
        if neighbourhood[0][0] == 1 and neighbourhood[2][2] == 1:
            result = True
        elif neighbourhood[0][1] == 1 and neighbourhood[2][1] == 1:
            result = True
        elif neighbourhood[1][0] == 1 and neighbourhood[1][2] == 1:
            result = True
        elif neighbourhood[0][2] == 1 and neighbourhood[2][0] == 1:
            result = True

    return result


def init_contour(point, neighbourhood):
    """
    :param point: row,col point in image
    :param neighbourhood: 3x3 patch of edge image THAT CONTAINS A CLEAN LINE (see above)
    :return: part of a contour, a list of points that forms line around the point,
        within its 3x3 neighbourhood
    """
    if not has_clean_line(neighbourhood):
        raise Exception("Need neighbourhood with clean line to initialize")

    contour = []
    if neighbourhood[0][0] == 1:
        contour.append((point[0]-1, point[1]-1))
        contour.append((point[0], point[1]))
        contour.append((point[0]+1, point[1]+1))
    elif neighbourhood[0][1] == 1:
        contour.append((point[0]-1, point[1]))
        contour.append((point[0], point[1]))
        contour.append((point[0]+1, point[1]))
    elif neighbourhood[0][2] == 1:
        contour.append((point[0]-1, point[1]+1))
        contour.append((point[0], point[1]))
        contour.append((point[0]+1, point[1]-1))
    elif neighbourhood[1][0] == 1:
        contour.append((point[0], point[1]-1))
        contour.append((point[0], point[1]))
        contour.append((point[0], point[1]+1))
    return contour


def angle_between_points(p1, p2):
    return np.arctan2(p2[1]-p1[1], p2[0]-p1[0])


def difference_of_angles(a1, a2):
    d = a1 - a2
    if d < -np.pi:
        d += 2*np.pi
    if d > np.pi:
        d -= 2*np.pi
    return d


def extend(image, contour):
    """
    :param image: 2D array of edge (1) and non-edge (0) pixels
    :param contour: list of points that makes up part of a contour
    :return: True if the contour keeps going in this direction (can extend further)
    """
    pa = contour[-2]
    pb = contour[-1]
    drow = pb[0] - pa[0]
    dcol = pb[1] - pa[1]
    if drow == 0:
        candidates = [(pb[0], pb[1]+dcol), (pb[0]-1, pb[1]+dcol), (pb[0]+1, pb[1]+dcol)]
    elif dcol == 0:
        candidates = [(pb[0]+drow, pb[1]), (pb[0]+drow, pb[1]-1), (pb[0]+drow, pb[1]+1)]
    else:
        candidates = [(pb[0]+drow, pb[1]+dcol), (pb[0]+drow, pb[1]), (pb[0], pb[1]+dcol)]

    contour_angle = angle_between_points(contour[-3], contour[-1])

    def candidate_valid(c):
        return c[0] >= 0 and c[0] < image.shape[0] and c[1] >= 0 and c[1] < image.shape[1]

    best_difference = np.pi
    best_candidate = None
    for candidate in candidates:
        if candidate_valid(candidate):
            candidate_angle = angle_between_points(contour[-2], candidate)
            difference = abs(difference_of_angles(candidate_angle, contour_angle))
            if difference < best_difference and image[candidate[0]][candidate[1]] == 1:
                best_difference = difference
                best_candidate = candidate

    keep_going = False
    if best_candidate is not None:
        contour.append(best_candidate)
        keep_going = True

        # stop if there has been a sharp turn recently
        if len(contour) >= 8:
            last_angle = angle_between_points(contour[-4], contour[-1])
            previous_angle = angle_between_points(contour[-8], contour[-5])
            if difference_of_angles(last_angle, previous_angle) >= np.pi/4:
                keep_going = False

    return keep_going


def show_contour(image, contour):
    for point in contour:
        image[point[0], point[1]] = .5
    plt.imshow(image)
    print(contour)
    # plt.show()


def get_random_contour(image, show=False):
    done = False
    while not done:
        point = get_random_point_on_an_edge(image)
        n = get_neighbourhood(image, point)

        if has_clean_line(n):
            contour = init_contour(point, n)

            ok = True
            while ok:
                ok = extend(image, contour)

            # go the other way
            contour = list(reversed(contour))
            ok = True
            while ok:
                ok = extend(image, contour)

            if len(contour) > 30:
                show_contour(image, contour)
                done = True

    return contour


# ---------------------------------------------------------------------------------------
plt.ion()

# hard-coded image (part of image)
filename = 'data/BIPED/edges/edge_maps/test/rgbr/RGB_008.png'

for idx in range(10):
    image = img.imread(filename)
    # image = image[:300,:300]
    print("Contour Length {}".format(len(get_random_contour(image, show=True))))
    pdb.set_trace()

# ---------------------------------------------------------------------------------------
print("End")
pdb.set_trace()