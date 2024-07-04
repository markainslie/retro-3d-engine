"""Retro-ish 3D renderer to re-refresh/improve my understanding of 3D concepts using PyGame. 
Excuse the mess...
"""

import pygame as pg
from primitives import Camera, Light, Mesh


class RenderScene:
    """Render a simple 3D object."""

    def __init__(self):

        # create params for game window
        self.res = self.res_x, self.res_y = 640, 480
        self.aspect_ratio = self.res_x / self.res_y
        self.fps = 60

        # init PyGame
        pg.init()
        self.win = pg.display.set_mode(self.res)
        self.clock = pg.time.Clock()
        self.bg_color = (64, 64, 64, 255)

        # build the scene and run the main game loop
        self.build_scene()
        self.main_loop()

    def build_scene(self):
        """Put together all the elements in the scene."""

        # camera
        self.camera = Camera(self)
        self.camera.focal_length = 1.0
        self.camera.near = 1.0
        self.camera.far = 100.0
        self.camera.move()

        # light
        self.light = Light(self)
        self.light.rotate = [-45, 90, 0]
        self.light.move()

        # 3d mesh
        self.mesh = Mesh(self)
        self.mesh.import_geo("3d_models/teapot.obj")

        self.mesh.diffuse = [0.25, 0.75, 0.1]
        self.mesh.ambient = [0.5, 0.5, 0.5]

        self.mesh.translate = [0, 0, -50]
        self.mesh.rotate = [20, 0, 0]
        self.mesh.move()

    def main_loop(self):
        """Main loop"""

        running = True
        while running:
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    running = False

            # draw our object
            self.win.fill(pg.Color(self.bg_color))
            self.mesh.draw(mode="faces", normals=False)

            # refresh the screen
            pg.display.set_caption(str(self.clock.get_fps()))
            pg.display.flip()

            # check the frame rate
            self.clock.tick(self.fps)

            # move stuff
            self.mesh.rotate[1] += 1
            self.mesh.move()


if __name__ == "__main__":
    app = RenderScene()
