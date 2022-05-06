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
        print(i)
        for j in range(height):
            ray = construct_ray_through_pixel(camera, (i/width-0.5)*camera.screen_width, (j/height-0.5)*camera.screen_height)
            intersection_point, intersected_shape = find_closest_intersection(ray, shapes)
            img[height-1-j][i] = get_color(intersection_point, intersected_shape, ray, set_params, materials, lights, shapes)

    return img


def construct_ray_through_pixel(camera, w, h):
    towards = camera.look_at - camera.position
    towards = towards/np.linalg.norm(towards)

    up_perp = camera.up - np.dot(camera.up,towards)/np.dot(towards,towards)*towards
    up_perp = up_perp/np.linalg.norm(up_perp)

    # width_direction = np.cross(towards, up_perp)
    width_direction = np.cross(up_perp, towards)
    width_direction = width_direction/np.linalg.norm(width_direction)

    target_point = camera.position + towards*camera.screen_dist + up_perp*h + width_direction*w
    ray_direction = target_point - camera.position

    return Ray(origin=camera.position,
               direction=ray_direction)


def find_closest_intersection(ray, shapes):
    best_intersection = None
    best_shape = None
    for shape in shapes:
        current_intersection = shape.find_intersection(ray)
        if current_intersection is not False:
            if best_intersection is None:
                best_intersection = current_intersection
                best_shape = shape
            else:
                if np.linalg.norm(ray.origin-best_intersection) > np.linalg.norm(ray.origin-current_intersection):
                    best_intersection = current_intersection
                    best_shape = shape
    return best_intersection, best_shape


def get_color(intersection_point, intersected_shape, camera_ray, set_params, materials, lights, shapes):
    if intersection_point is None:
        return set_params.background_rgb

    current_material = materials[intersected_shape.material-1]

    color_out = np.array([0.0, 0.0, 0.0])

    for light in lights:
        light_direction = intersection_point - light.position
        light_direction = light_direction/np.linalg.norm(light_direction)
        light_ray = Ray(origin=light.position, direction=light_direction)
        light_intersection_point, light_intersected_shape = find_closest_intersection(light_ray, shapes)

        # skip if the light does not reach the intersection point
        if light_intersection_point is None:
            continue
        if not np.all(np.isclose(light_intersection_point, intersection_point)):
            continue

        # diffuse coloring
        surface_normal = intersected_shape.normal_at_point(light_intersection_point)
        color_out += current_material.diffuse_rgb * np.abs(np.dot(surface_normal, -light_direction))

        # specular coloring
        # reflect_direction calculation: shading13.pdf - slide 41 "The Highlight Vector"
        reflect_direction = light_direction - 2*np.dot(light_direction, surface_normal)*surface_normal
        color_out += current_material.specular_rgb * np.power(np.abs(np.dot(reflect_direction, -camera_ray.direction)), current_material.phong)

    # TODO: is this ok?
    color_out[color_out > 1] = 1

    return color_out

if __name__ == '__main__':
    main()
