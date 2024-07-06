"""Functions for 3D transform matrix stuff.
"""

import numpy as np


def translate(pos):
    """Translate XYZ matrix

    Args:
        pos (list): 3D [x, y, z] vector

    Returns:
        numpy.array: 4x4 matrix
    """

    return np.array(
        [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [pos[0], pos[1], pos[2], 1],
        ]
    )


def rotate_x(a):
    """Rotate X matrix

    Args:
        a (float): angle in radians

    Returns:
        numpy.array: 4x4 matrix
    """

    return np.array(
        [
            [1, 0, 0, 0],
            [0, np.cos(a), np.sin(a), 0],
            [0, -np.sin(a), np.cos(a), 0],
            [0, 0, 0, 1],
        ]
    )


def rotate_y(a):
    """Rotate Y matrix

    Args:
        a (float): angle in radians

    Returns:
        numpy.array: 4x4 matrix
    """

    return np.array(
        [
            [np.cos(a), 0, -np.sin(a), 0],
            [0, 1, 0, 0],
            [np.sin(a), 0, np.cos(a), 0],
            [0, 0, 0, 1],
        ]
    )


def rotate_z(a):
    """Rotate Z matrix

    Args:
        a (float): angle in radians

    Returns:
        numpy.array: 4x4 matrix
    """

    return np.array(
        [
            [np.cos(a), np.sin(a), 0, 0],
            [-np.sin(a), np.cos(a), 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ]
    )


def scale(scl):
    """Scale XYZ matrix

    Args:
        scl (list): 3D [x, y, z] vector

    Returns:
        numpy.array: 4x4 matrix
    """

    return np.array(
        [
            [scl[0], 0, 0, 0],
            [0, scl[1], 0, 0],
            [0, 0, scl[2], 0],
            [0, 0, 0, 1],
        ]
    )


def get_rotation_matrix(rot_vect, rot_order="xyz", inv=False):
    """Convert euler rotate x, y, z coords to a 4x4 matrix.

    Args:
        rot_vect (list): 3D euler rotate coords
        rot_order (str, optional):  Maya-like rotate order. Defaults to "xyz".
        inv (bool, optional): Inverts the resulting matrix. Defaults to False.

    Returns:
        numpy.array: 4x4 matrix
    """

    # convert degrees to radians
    rot = list()
    for angle in rot_vect[:3]:
        rot.append(np.radians(angle))

    # define each rotation matrix
    rot_x_matrix = rotate_x(rot[0])
    rot_y_matrix = rotate_y(rot[1])
    rot_z_matrix = rotate_z(rot[2])

    # dot product all the rotation matrices in the desired order (default is xyz)
    if rot_order == "yzx":
        rot_matrix = rot_y_matrix @ rot_z_matrix @ rot_x_matrix
    elif rot_order == "zxy":
        rot_matrix = rot_z_matrix @ rot_x_matrix @ rot_y_matrix
    elif rot_order == "xzy":
        rot_matrix = rot_x_matrix @ rot_z_matrix @ rot_y_matrix
    elif rot_order == "yxz":
        rot_matrix = rot_y_matrix @ rot_x_matrix @ rot_z_matrix
    elif rot_order == "zyx":
        rot_matrix = rot_z_matrix @ rot_y_matrix @ rot_x_matrix
    else:  # xyz
        rot_matrix = rot_x_matrix @ rot_y_matrix @ rot_z_matrix

    # for inverse-sake
    if inv:
        rot_matrix = np.linalg.inv(rot_matrix)

    return rot_matrix


def get_transform_matrix(pos_vect, rot_vect, rot_order="xyz", inv=False):
    """Convert translate and rotate x, y, z coords to a 4x4 matrix.

    Args:
        pos_vect (list): translate coords
        rot_vect (list): 3d euler rotate coords
        rot_order (str, optional): Maya-like rotate order. Defaults to "xyz".
        inv (bool, optional): Inverts the resulting matrix. Defaults to False.

    Returns:
        numpy.array: 4x4 matrix
    """

    # dot product the crap out of everything...
    rot_matrix = get_rotation_matrix(rot_vect, rot_order, inv=False)
    pos_matrix = translate(pos_vect)
    out_matrix = rot_matrix @ pos_matrix

    # in case we need the inverse
    if inv:
        out_matrix = np.linalg.inv(out_matrix)

    return out_matrix
