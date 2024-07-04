"""Functions for vectors.
"""

import numpy as np


def normalize_vector(vect):
    """Scale vector magnitude to 1

    Args:
        vect (list): vector

    Returns:
        list: normalized vector
    """

    vect = np.array(vect[:3])
    vect = vect / np.linalg.norm(vect)

    return vect.tolist()


def average_vectors(vect_list):
    """Take a list of vertices and average them to find the centroid.

    Args:
        vect_list (list): list of 3D vectors (vertices)

    Returns:
        list: centroid of the vertices
    """

    vect_list = np.array([v for v in vect_list])
    vect = sum(vect_list) / len(vect_list)

    return vect.tolist()


def get_triangle_normal(v1, v2, v3):
    """Get the normal of a triangle. The order of the vertices is important.

    Args:
        v1 (list): first vertex
        v2 (list): second vertex
        v3 (list): third vertex

    Returns:
        list: direction vector
    """

    # convert all to numpy arrays
    v1 = np.array(v1[:3])
    v2 = np.array(v2[:3])
    v3 = np.array(v3[:3])

    return normalize_vector(np.cross(v2 - v1, v3 - v1))


def get_polygon_normal(vertex_list):
    """Get the face normal of a 3 or 4 point polygon.

    Args:
        vertex_list (list): list of 3d vertices

    Returns:
        list: direction vector
    """

    normal_list = [get_triangle_normal(vertex_list[0], vertex_list[1], vertex_list[2])]
    if len(vertex_list) == 4:
        normal_list.append(
            get_triangle_normal(vertex_list[2], vertex_list[3], vertex_list[1])
        )

    return normalize_vector(average_vectors(normal_list))
