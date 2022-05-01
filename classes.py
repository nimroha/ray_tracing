from dataclasses import dataclass

import numpy as np


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
