from dataclasses import dataclass

import numpy as np

from utils import norm2

TANGENT_TOLERANCE = 0.01

@dataclass
class Ray:
    origin:    np.ndarray
    direction: np.ndarray

    def __post_init__(self):
        """normalize direction"""
        self.direction /= norm2(self.direction)

    def project(self, point):
        """
        projection of the `direction` vector on the `origin->`point` vector
        :param point: (x,y,z) numpy array
        :return: length of projection
        """
        return np.dot(self.direction, point - self.origin)

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

    def normal_at_point(self, point: np.array):
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
            return False

        oc = c - ray.origin
        oc_size = norm2(oc)
        cb_size = np.sqrt(oc_size ** 2 - ob_size ** 2)
        # if np.isclose(cb_size, r, rtol=TANGENT_TOLERANCE):
        if np.abs(cb_size - r) <= TANGENT_TOLERANCE:
            return ray.get_point(ob_size)

        if cb_size > r:
            return False

        t_diff_abs = np.sqrt(r ** 2 - cb_size ** 2)
        p1 = ray.get_point(ob_size - t_diff_abs)
        p2 = ray.get_point(ob_size + t_diff_abs)
        if np.linalg.norm(ray.origin-p1) < np.linalg.norm(ray.origin-p2):
            return p1
        else:
            return p2

    def normal_at_point(self, point: np.array):
        normal_vec = point-self.center
        normal_vec = normal_vec/np.linalg.norm(normal_vec)
        return normal_vec

@dataclass
class Plane(Shape):
    normal: np.ndarray
    offset: float

    def __post_init__(self):
        """normalize normal"""
        self.normal /= norm2(self.normal)

    def find_intersection(self, ray: Ray):
        """
        find the ray's intersection with a plane
        denote:
        o - ray origin
        d - ray direction
        n - plane normal
        c - plane offset
        p - the intersection point
        t - the respective line parameter

        use the plane equation <n,p> = <n,o + t * d> = <n,o> + t * <n,d> = c to find the intersection point (if exists)

        :param ray: a Ray instance
        :return: the intersecting point, or False for no intersection
        """
        n_dot_d = np.dot(self.normal, ray.direction)
        # if np.isclose(n_dot_d, 0, rtol=0.01):
        if np.abs(n_dot_d) < 0.01:
            return False # ray is parallel to the plane

        n_dot_o = np.dot(self.normal, ray.origin)
        t = (self.offset - n_dot_o) / n_dot_d
        point = ray.get_point(t)

        return point

    def normal_at_point(self, point: np.array):
        return self.normal

@dataclass
class Box(Shape):
    center: np.ndarray
    length: float

    def find_intersection(self, ray: Ray):
        half_length = self.length / 2
        box_min = self.center - half_length
        box_max = self.center + half_length
        t_min = np.divide(box_min - ray.origin, ray.direction)
        t_max = np.divide(box_max - ray.origin, ray.direction)
        # TODO we can probably return False her if no t is positive

        # if we exit the slab on any axis before entering on all axes, then no intersection
        if (t_min[0] > min(t_max[1], t_max[2]) or
            t_min[1] > min(t_max[0], t_max[2]) or
            t_min[2] > min(t_max[0], t_max[1])):
            return False

        # the minimum t inside all 3 slabs, is the max of mins # TODO make sure no edge cases
        t = t_min.max()
        return ray.get_point(t)
