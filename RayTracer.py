import argparse
import numpy as np

from parser import parse
from utils import write_img
from utils import is_close
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
            intersection_point, intersected_shape, intersected_shape_index = find_closest_intersection(ray, shapes)
            img[height-1-j][i], _ = get_color(intersection_point, intersected_shape, intersected_shape_index, ray, set_params, materials, lights, shapes)

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
    best_index = None
    index = 0
    for shape in shapes:
        current_intersection = shape.find_intersection(ray)
        if current_intersection is not False:
            if np.dot(current_intersection-ray.origin, ray.direction) < 0:
                # point is behind the ray
                continue
            if best_intersection is None:
                best_intersection = current_intersection
                best_shape = shape
                best_index = index
            else:
                if np.linalg.norm(ray.origin-best_intersection) > np.linalg.norm(ray.origin-current_intersection):
                    best_intersection = current_intersection
                    best_shape = shape
                    best_index = index
        index += 1
    return best_intersection, best_shape, best_index


def get_color(intersection_point, intersected_shape, intersected_shape_index, camera_ray, set_params, materials, lights, shapes):
    if intersection_point is None:
        return set_params.background_rgb, True

    current_material = materials[intersected_shape.material-1]

    color_out = np.array([0.0, 0.0, 0.0])
    shapes_without_current_object = None

    for light in lights:
        light_direction = intersection_point - light.position
        light_direction = light_direction/np.linalg.norm(light_direction)
        light_ray = Ray(origin=light.position, direction=light_direction)
        light_intersection_point, light_intersected_shape, light_intersected_shape_index = find_closest_intersection(light_ray, shapes)
        light_hit_different_object = False

        if light_intersection_point is None:
            # skip if the light does not reach any object
            continue
        if not np.all(is_close(light_intersection_point, intersection_point)):
            if light_intersected_shape_index == intersected_shape_index:
                # skip if the light hit the other side of the shape
                continue
            light_hit_different_object = True

        # diffuse coloring
        surface_normal = intersected_shape.normal_at_point(light_intersection_point)
        diffuse_color = current_material.diffuse_rgb * np.abs(np.dot(surface_normal, -light_direction))

        # specular coloring
        # reflect_direction calculation: shading13.pdf - slide 41 "The Highlight Vector"
        reflect_direction = light_direction - 2*np.dot(light_direction, surface_normal)*surface_normal
        specular_color = current_material.specular_rgb * np.power(np.abs(np.dot(reflect_direction, -camera_ray.direction)), current_material.phong) * light.specular_intens

        # soft shadows
        # perc_rays_hit = get_soft_shadow_perc_rays_hit(light_ray, set_params.root_shadow_rays, light.radius, intersection_point, shapes)
        # light_intensity = (1-light.shadow_intens)*1 + light.shadow_intens*perc_rays_hit
        light_intensity = 1

        # transparency
        if current_material.transp > 0:
            if shapes_without_current_object is None:
                if len(shapes) == 0:
                    shapes_without_current_object = shapes
                else:
                    shapes_without_current_object = shapes.copy()
                    del shapes_without_current_object[light_intersected_shape_index]

            inner_intersection_point, inner_intersected_shape, inner_intersected_shape_index = find_closest_intersection(camera_ray, shapes_without_current_object)
            back_color, hit_background = get_color(inner_intersection_point, inner_intersected_shape, inner_intersected_shape_index, camera_ray, set_params, materials, lights, shapes_without_current_object)
            if not hit_background:
                back_color = back_color * light.rgb
        else:
            back_color = np.array([0.0, 0.0, 0.0])

        # light intensity only affects the diffuse and specular colors
        cur_light_color_out = light.rgb*light_intensity*(diffuse_color+specular_color)*(1-current_material.transp) + current_material.transp*back_color
        if light_hit_different_object:
            color_out += cur_light_color_out*(1-light.shadow_intens)
        else:
            color_out += cur_light_color_out

    color_out[color_out > 1] = 1

    return color_out, False


def get_soft_shadow_perc_rays_hit(light_ray, num_shadow_rays, radius, intersection_point, shapes):
    # Find a plane which is perpendicular to the ray
    axis_one = np.array([0,0,1])
    if np.all(is_close(axis_one, light_ray.direction)):
        axis_one = np.array([1, 0, 0])

    axis_one = axis_one - np.dot(axis_one, light_ray.direction) / np.dot(light_ray.direction, light_ray.direction) * light_ray.direction
    axis_one = axis_one / np.linalg.norm(axis_one)
    axis_two = np.cross(axis_one, light_ray.direction)
    axis_two = axis_two / np.linalg.norm(axis_two)

    light_hit_counter = 0

    # Define a rectangle & divide the rectangle into a grid of N x N cells
    rand_width = radius / num_shadow_rays
    for i in np.linspace(-radius/2+rand_width/2,radius/2-rand_width/2,num_shadow_rays):
        for j in np.linspace(-radius/2+rand_width/2,radius/2-rand_width/2,num_shadow_rays):
            #  we select a random point in each cell (by uniformly sampling x value and y value)
            i_perturbation = i + np.random.uniform() * rand_width - rand_width / 2
            j_perturbation = j + np.random.uniform() * rand_width - rand_width / 2

            # perturbation cell center calculation
            current_cell_center = light_ray.origin + axis_one * i_perturbation + axis_two * j_perturbation

            light_direction = intersection_point - current_cell_center
            light_direction = light_direction / np.linalg.norm(light_direction)
            light_ray = Ray(origin=current_cell_center, direction=light_direction)
            light_intersection_point, light_intersected_shape, _ = find_closest_intersection(light_ray, shapes)

            # skip if the light does not reach the intersection point
            if light_intersection_point is None:
                continue
            if not np.all(is_close(light_intersection_point, intersection_point)):
                continue

            light_hit_counter += 1

    return light_hit_counter/(num_shadow_rays*num_shadow_rays)

if __name__ == '__main__':

    main()
