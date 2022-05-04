import argparse
import numpy as np

from parser import parse
from utils import write_img
from classes import Camera, Set, Material, Light, Sphere, Plane, Box, Ray




def main():
    parser = argparse.ArgumentParser(description='render a 3D scene to a 2D image')
    parser.add_argument('scene',   metavar='scene',   help='The input scene definition path', type=str)
    parser.add_argument('output',  metavar='output',  help='The output image path',           type=str)
    parser.add_argument('width',   metavar='width',   help='The output image path width',     type=int, nargs='?', default=500)
    parser.add_argument('height',  metavar='height',  help='The output image path height',    type=int, nargs='?', default=500)
    args = parser.parse_args()

    print(args)
    camera, set_params, materials, lights, shapes = parse(args.scene,args.width,args.height)

    img = ray_cast(args.height, args.width, camera, set_params, materials, lights, shapes)

    write_img(img, args.output)


def ray_cast(height, width, camera, set_params, materials, lights, shapes):
    img = np.zeros([height, width, 3], dtype=np.float32)  # converted to uint8 before saving

    for i in range(width):
        for j in range(height):
            ray = construct_ray_through_pixel(camera, (i/width-0.5)*camera.screen_width, (j/height-0.5)*camera.screen_height)
            # Intersection hit = FindIntersection(ray, scene);
            # img[i][j] = GetColor(hit);

    return img


def construct_ray_through_pixel(camera, w, h):
    towards = camera.look_at - camera.position
    towards = towards/np.linalg.norm(towards)

    up_perp = camera.up - np.dot(camera.up,towards)/np.dot(towards,towards)*towards
    up_perp = up_perp/np.linalg.norm(up_perp)

    width_direction = np.cross(towards, up_perp)
    # width_direction = np.cross(up_perp, towards) # TODO: check
    width_direction = width_direction/np.linalg.norm(width_direction)

    target_point = camera.position + towards*camera.screen_dist + up_perp*h + width_direction*w
    ray_direction = target_point - camera.position
    ray_direction = ray_direction/np.linalg.norm(ray_direction)

    return Ray(origin=camera.position,
               direction=ray_direction)


if __name__ == '__main__':
    main()
