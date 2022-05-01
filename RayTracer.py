import argparse
import numpy as np

from dataclasses import dataclass


@dataclass
class Camera:
    position:     np.ndarray
    look_at:      np.ndarray
    up:           np.ndarray
    screen_dist:  float
    screen_width: float

@dataclass
class Set:
    background_rgb:   np.ndarray
    root_shadow_rays: int
    max_recursions:   int

@dataclass
class Material:
    diffuse_rgb:  np.ndarray
    specular_rgb: np.ndarray
    reflect_rgb:  np.ndarray
    phong:        float
    transp:       float

@dataclass
class Light:
    position:        np.ndarray
    rgb:             np.ndarray
    specular_intens: float
    shadow_intens:   float
    radius:          float

@dataclass
class Shape:
    material: int

    def find_intersection(self, ray):
        # will raise if we miss an implementation
        raise NotImplementedError(f'the subclass {self.__class__} did not implement this method')


@dataclass
class Sphere(Shape):
    center: np.ndarray
    radius: float


@dataclass
class Plane(Shape):
    normal: np.ndarray
    offset: float

@dataclass
class Box(Shape):
    center: np.ndarray
    length: float


def parse(scene_path):
    with open(scene_path, 'r') as fp:
        materials = []
        lights    = []
        shapes    = []
        for line in fp.readlines():
            line = line.strip()
            if len(line) == 0 or line.startswith('#'):
                continue

            code   = line[:3]
            values = [float(v) for v in line[3:].split()]
            print(code, values)
            if code == 'cam':
                camera = Camera(position=np.array(values[0:3]),
                                look_at=np.array(values[3:5]),
                                up=np.array(values[5:8]),
                                screen_dist=values[9],
                                screen_width=values[10])
            elif code == 'set':
                set_params = Set(background_rgb=np.array(values[0:3]),
                                 root_shadow_rays=int(values[3]),
                                 max_recursions=int(values[4]))
            elif code == 'mtl':
                materials.append(Material(diffuse_rgb=np.array(values[0:3]),
                                          specular_rgb=np.array(values[3:5]),
                                          reflect_rgb=np.array(values[5:8]),
                                          phong=values[9],
                                          transp=values[10]))
            elif code == 'sph':
                shapes.append(Sphere(center=np.array(values[0:3]),
                                     radius=values[3],
                                     material=int(values[4])))
            elif code == 'pln':
                shapes.append(Plane(normal=np.array(values[0:3]),
                                    offset=values[3],
                                    material=int(values[4])))
            elif code == 'box':
                shapes.append(Box(center=np.array(values[0:3]),
                                  length=values[3],
                                  material=int(values[4])))
            elif code == 'lgt':
                lights.append(Light(position=np.array(values[0:3]),
                                    rgb=np.array(values[3:5]),
                                    specular_intens=values[6],
                                    shadow_intens=values[7],
                                    radius=values[8]))

    return camera, set_params, materials, lights, shapes


def main():
    parser = argparse.ArgumentParser(description='render a 3D scene to a 2D image')
    parser.add_argument('scene',   metavar='scene',   help='The input scene definition path', type=str)
    parser.add_argument('output',  metavar='output',  help='The output image path',           type=str)
    parser.add_argument('width',   metavar='width',   help='The output image path width',     type=int, nargs='?', default=500)
    parser.add_argument('height',  metavar='height',  help='The output image path height',    type=int, nargs='?', default=500)
    args = parser.parse_args()

    print(args)
    parsed = parse(args.scene)
    print(parsed)

if __name__ == '__main__':
    main()
