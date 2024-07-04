"""Functions for 3D projections.
"""

from numba import njit


@njit()
def project_points(points, focal_length, aspect_ratio, near, far):
    """Projects a list of 3D points onto the screen. This resolved to normalized display
    coordinates. In other words, X and Z range from -1.0 to 1.0, while Z ranges from 0.0 
    (near) to 1.0 (far) of the cliping plane. So, coords [0, 0, 0] is in the center of the 
    screen on the near clipping plane. Get it?

    I'm sure there's a fancy matrixy way of doing this, but brain is full.

    Args:
        points (list): A list of 3D points in camera space
        focal_length (float): Focal length of the camera
        aspect_ratio (float): Aspect ratio of the display
        near (float): Near clipping plane
        far (_type_): Far clipping plane

    Returns:
        list: A list of NDC-ish pixel values
    """

    # we only need to calulate this one time
    sz = 1 / (far - near)

    # iterate over the points and project them to screen space
    out_points = []
    for pos in points:
        tx, ty, tz = pos[:3]  # limit the vector to an array of 3 elements

        # define our scaling factor
        sx = focal_length / abs(tz)
        sy = sx * aspect_ratio

        # plot out the points in to NDC space
        px = tx * (focal_length / abs(tz))
        py = ty * sy
        pz = (-tz - near) * sz

        out_points.append([px, py, pz])

    return out_points
