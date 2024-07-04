"""Meat and potatoes of creating primitives for a 3D scene.
TODO: Reduce the clutter. Mary Kondo would be sad...
"""

import os
import re
import numpy as np
import pygame as pg

from projection_utils import project_points
from vector_utils import average_vectors, get_polygon_normal
from matrix_utils import get_transform_matrix, get_rotation_matrix


class Prim:
    """Base primitive class"""

    def __init__(self, render):
        self.render = render

        # transform information
        self.translate = [0.0, 0.0, 0.0]
        self.rotate = [0.0, 0.0, 0.0]
        # self.scale = [1.0, 1.0, 1.0]

        self.matrix = None
        self.inverse_matrix = None

    def move(self):
        """Move the primitive by updating the matrices."""
        self.matrix = get_transform_matrix(
            self.translate, self.rotate, rot_order="xyz", inv=False
        )
        self.inverse_matrix = get_transform_matrix(
            self.translate, self.rotate, rot_order="xyz", inv=True
        )


class Camera(Prim):
    """Basic camera class"""

    def __init__(self, render):
        super().__init__(render)

        # camera info
        self.focal_length = 0.50
        self.sensor_width = 0.36
        self.sensor_height = self.sensor_width / self.render.aspect_ratio

        self.near = 1.0
        self.far = 10.0

    def move(self):
        self.matrix = get_transform_matrix(
            self.translate, self.rotate, rot_order="zxy", inv=False
        )
        self.inverse_matrix = get_transform_matrix(
            self.translate, self.rotate, rot_order="zxy", inv=True
        )


class Light(Prim):
    """Basic light class"""

    def __init__(self, render):
        super().__init__(render)

        self.color = np.array([1.0, 1.0, 1.0])
        # self.ambient_color = [0.1, 0.1, 0.1]

        self.move()

    def move(self):
        light_vector = np.array([0, 0, -1, 1])
        self.light_vector = (
            light_vector
            @ get_rotation_matrix(self.rotate, rot_order="xyz")
            @ get_rotation_matrix(self.render.camera.rotate, rot_order="zxy", inv=True)
        )


class Mesh(Prim):
    """Mesh class that loads a .obj file"""

    def __init__(self, render, data=None):
        super().__init__(render)

        self.diffuse = np.array([0.5, 0.5, 0.5])
        self.ambient = np.array([0.3, 0.3, 0.3])

        # shape information
        if data:
            self.data = data
        else:
            self.init_data()
            self.import_geo("3d_models/cube.obj")
            self.move()

    @property
    def diffuse(self):
        return self._diffuse

    @diffuse.setter
    def diffuse(self, value):
        self._diffuse = np.array(value)

    @property
    def ambient(self):
        return self._ambient

    @ambient.setter
    def ambient(self, value):
        self._ambient = np.array(value)

    def init_data(self):
        """Initialize the geo data dictionary."""
        self.data = {
            "f": [],
            "vert_num": 0,
            "face_num": 0,
            "v": np.array([]),
            "vn": np.array([]),
            "fcv": np.array([]),
            "fn": np.array([]),
            "fdn": np.array([]),
        }

    def move(self):
        self.matrix = get_transform_matrix(
            self.translate, self.rotate, rot_order="xyz", inv=False
        )
        self.inverse_matrix = get_transform_matrix(
            self.translate, self.rotate, rot_order="xyz", inv=True
        )

        out_transform_matrix = self.matrix @ self.render.camera.inverse_matrix

        self.cam_v = self.data.get("v") @ out_transform_matrix
        self.cam_fcv = self.data.get("fcv") @ out_transform_matrix
        self.cam_fdn = self.data.get("fdn") @ out_transform_matrix

        # rotate only direction vectors
        world_rotate_matrix = get_rotation_matrix(self.rotate, rot_order="xyz")
        cam_inverse_rotate_matrix = get_rotation_matrix(
            self.render.camera.rotate, rot_order="zxy", inv=True
        )
        out_rot_matrix = world_rotate_matrix @ cam_inverse_rotate_matrix

        # rotate normals to camera space
        self.cam_fn = self.data.get("fn") @ out_rot_matrix

        # transform renderable points to display space
        self.screen_v = project_points(
            self.cam_v,
            self.render.camera.focal_length,
            self.render.aspect_ratio,
            self.render.camera.near,
            self.render.camera.far,
        )
        self.screen_fcv = project_points(
            self.cam_fcv,
            self.render.camera.focal_length,
            self.render.aspect_ratio,
            self.render.camera.near,
            self.render.camera.far,
        )
        self.screen_fdn = project_points(
            self.cam_fdn,
            self.render.camera.focal_length,
            self.render.aspect_ratio,
            self.render.camera.near,
            self.render.camera.far,
        )

    def import_geo(self, path):
        """Import an .obj file.

        Args:
            path (string): Path to the .obj file

        Raises:
            FileNotFoundError: If we can't find a file, barf!
        """

        # check for an existing path
        if not os.path.exists(path):
            raise FileNotFoundError(path)

        f_list = []  # face list derived from file
        v_list = []  # vertex list derived from file
        vn_list = []  # vertex normal list derived from file
        fcv_list = []  # computed face centroid vertex
        fn_list = []  # computed face normal
        fdn_list = []  # computed face display normal (centroid + normal)

        # parse the obj file
        with open(path, encoding="utf-8") as file:
            while line := file.readline():
                ln = re.sub(" +", " ", line.rstrip()).split(" ")
                if ln[0] == "v":
                    v_list.append([float(ln[1]), float(ln[2]), float(ln[3]), 1])
                if ln[0] == "vn":
                    vn_list.append([float(ln[1]), float(ln[2]), float(ln[3]), 1])
                if ln[0] == "f":
                    f_list.append([int(f.split("/")[0]) - 1 for f in ln[1:]])
                    # fvn_list.append([int(f.split('/')[1])-1 for f in ln[1:]])

        # go over faces again and compute extra attributes
        for face in f_list:
            face_verts = [v_list[vnum] for vnum in face]
            fcv = average_vectors(face_verts)
            fn = get_polygon_normal(face_verts)
            fn = [fn[0], fn[1], fn[2], 1]
            fdn = np.add(fcv[:3], fn[:3])
            fdn = [fdn[0], fdn[1], fdn[2], 1]

            fcv_list.append(fcv)
            fn_list.append(fn)
            fdn_list.append(fdn)

        self.data["f"] = f_list  # faces
        self.data["v"] = np.array(v_list)  # vertices
        self.data["vn"] = np.array(vn_list)  # vertex normals

        self.data["fcv"] = np.array(fcv_list)  # face centroid vertices - calculated
        self.data["fn"] = np.array(fn_list)  # face normals - calculated
        self.data["fdn"] = np.array(fdn_list)  # face display normals - calculated

        self.data["face_num"] = len(f_list)
        self.data["vert_num"] = len(v_list)

    def draw(self, mode="points", normals=False):
        """Draw the 3D object to the screen

        Args:
            mode (str, optional): Draw mode: "points, faces". Defaults to "points".
            normals (bool, optional): Shows normals in "faces" mode. Defaults to False.
        """

        # draw points to screen
        if mode == "points":
            for vert in self.screen_v:
                # skip the point if it sits outside the clipping planes
                if vert[2] < 0 or vert[2] > 1:
                    continue

                # draw our pixels
                pixel_x = (vert[0] + 0.5) * self.render.res_x
                pixel_y = self.render.res_y - ((vert[1] + 0.5) * self.render.res_y)
                pixel_z = 255 - int(vert[2] * 255)

                shad_val = pg.Color(pixel_z, pixel_z, pixel_z)
                pg.draw.circle(self.render.win, shad_val, [pixel_x, pixel_y], 2, 0)

        # draw faces
        else:
            # cull backfaces and out of frustrum faces
            face_list = []
            for fnum in range(self.data.get("face_num")):
                cull_backfaces = True

                # cast a ray from the face vertex to the camera and dot product with the face normal
                is_facing = 1
                if cull_backfaces:
                    is_facing = -self.cam_fcv[fnum][:3] @ self.cam_fn[fnum][:3]

                if (
                    is_facing > 0
                    and self.screen_fcv[fnum][2] < 1.0
                    and self.screen_fcv[fnum][2] > 0.0
                ):
                    face_list.append([self.screen_fcv[fnum][2], fnum])

            # start drawing faces from a sorted list (far to near)
            for _, fnum in sorted(face_list, key=lambda fc: fc[0], reverse=True):
                # calculate shading of the face (not terribly fancy or optimized)
                illum = self.cam_fn[fnum][:3] @ -self.render.light.light_vector[:3]
                illum_colour = np.clip(
                    (illum * self.render.light.color * self.diffuse)
                    + (self.diffuse * self.ambient),
                    0,
                    1,
                )
                shade = (illum_colour * 255).astype(np.int64)
                shad_val = pg.Color(shade)

                # draw our polygon
                pixel_list = []
                for vx, vy, _ in [
                    self.screen_v[vnum][:3] for vnum in self.data.get("f")[fnum]
                ]:
                    pixel_x = (vx + 0.5) * self.render.res_x
                    pixel_y = self.render.res_y - ((vy + 0.5) * self.render.res_y)
                    pixel_list.append([pixel_x, pixel_y])
                pg.draw.polygon(self.render.win, shad_val, pixel_list, 0)

                # draw the face centroid and face normal
                if normals:
                    su = (self.screen_fcv[fnum][0] + 0.5) * self.render.res_x
                    sv = self.render.res_y - (
                        (self.screen_fcv[fnum][1] + 0.5) * self.render.res_y
                    )
                    start_coords = [su, sv]

                    eu = (self.screen_fdn[fnum][0] + 0.5) * self.render.res_x
                    ev = self.render.res_y - (
                        (self.screen_fdn[fnum][1] + 0.5) * self.render.res_y
                    )
                    end_coord = [eu, ev]

                    line_col = pg.Color(255, 0, 0)
                    point_col = pg.Color(0, 0, 255)
                    pg.draw.circle(self.render.win, point_col, start_coords, 2, 0)
                    pg.draw.line(self.render.win, line_col, start_coords, end_coord, 1)
