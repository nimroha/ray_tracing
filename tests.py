import pytest
from classes import Camera, Set, Material, Light, Sphere, Plane, Box, Ray
from utils import norm2

import numpy as np

def test_intersection():
    ray = Ray(origin=np.array([0., 3., 0.]), direction=np.array([0.73723662, -0.67311481, 0.05829765]))
    sphere = Sphere(material=3, center=np.array([2., 0., 0.]), radius=1.0)

    point = sphere.find_intersection(ray)
    distance = norm2(point - sphere.center)
    assert np.isclose(distance, sphere.radius)

    tangent_ray = Ray(origin=np.array([3., -5., 0.]), direction=np.array([0., 1., 0.]))
    expected = np.array([3, 0, 0])
    assert np.all(sphere.find_intersection(tangent_ray) == expected)

    x_epsilon = [0.01, 0, 0]
    tangent_ray.origin += x_epsilon  # check tolerance
    assert np.all(sphere.find_intersection(tangent_ray) == (expected + x_epsilon))
