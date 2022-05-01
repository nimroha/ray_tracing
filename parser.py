import numpy as np

from classes import Camera, Set, Material, Light, Sphere, Plane, Box


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
