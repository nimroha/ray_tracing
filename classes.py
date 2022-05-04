from dataclasses import dataclass

import numpy as np

from utils import norm2

TANGENT_TOLERANCE = 0.01

@dataclass
class Ray:
    origin:    np.ndarray
    direction: np.ndarray

    def project(self, point):
        """
        projection of `point` vector on the `direction` vector
        :param point: (x,y,z) numpy array
        :return: length of projection
        """
        return np.dot(self.direction, point)

    def get_point(self, t):
        """
        parametrized point along the ray
        :param t: distance from origin
        :return: (x,y,z) numyp array
        """
        return self.origin + t * self.direction

@dataclass
class Camera:
    position:     np.ndarray
    look_at:      np.ndarray
    up:           np.ndarray
    screen_dist:  float
    screen_width: float
    screen_height: float


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

    def find_intersection(self, ray: Ray):
        # will raise if we miss an implementation
        raise NotImplementedError(f'the subclass {self.__class__} did not implement this method')


@dataclass
class Sphere(Shape):
    center: np.ndarray
    radius: float

    def find_intersection(self, ray: Ray):
        """
        find the ray's intersection with a sphere
        denote:
        o - ray origin
        d - ray direction
        c - sphere center
        r - sphere radius
        p1, p2 - intersection points
        t1, t2 - their respective line parameters
        a - the intersection of the ray with a perpendicular lin that passes through c

        use triangle building to find the intersection point(s), if exists

        :param ray: a Ray instance
        :return: the one or two intersecting points, or False for no intersection
        """
        c = self.center
        r = self.radius
        ob_size = ray.project(c)
        if ob_size <= 0:
            return False # TODO is this a good "no intersection" result?

        oc = c - ray.origin
        oc_size = norm2(oc)
        cb_size = np.sqrt(oc_size ** 2 - ob_size ** 2)
        if np.isclose(cb_size, r, rtol=TANGENT_TOLERANCE):
            return ray.get_point(ob_size)

        if cb_size > r:
            return False

        t_diff_abs = np.sqrt(r ** 2 - cb_size ** 2)
        return ray.get_point(cb_size - t_diff_abs), ray.get_point(cb_size + t_diff_abs)


@dataclass
class Plane(Shape):
    normal: np.ndarray
    offset: float

    def find_intersection(self, ray: Ray):
        """
        find the ray's intersection with a sphere
        denote:
        o - ray origin
        d - ray direction
        n - plane normal
        c - plane offset
        p - the intersection point
        t - the respective line parameter

        use the plane equation <n,o + t * d> = c to find the intersection point (if exists)

        :param ray: a Ray instance
        :return: the intersecting point, or False for no intersection
        """
        raise NotImplementedError('not finished')
        projection_on_normal = ray.project(self.normal)
        if projection_on_normal <= 0:
            return False # TODO is this a good "no intersection" result?

        t = (self.offset - np.dot(ray.origin, self.normal)) / projection_on_normal
        point = ray.get_point(t)

        return point


@dataclass
class Box(Shape):
    center: np.ndarray
    length: float

