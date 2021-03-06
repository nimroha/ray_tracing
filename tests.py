import pytest
from classes import Camera, Set, Material, Light, Sphere, Plane, Box, Ray
from utils import norm2

import numpy as np

def test_sphere_intersection():
    ray = Ray(origin=np.array([0., 3., 0.]), direction=np.array([0.73723662, -0.67311481, 0.05829765]))
    sphere = Sphere(material=3, center=np.array([2., 0., 0.]), radius=1.0)

    point = sphere.find_intersection(ray)
    distance = norm2(point - sphere.center)
    assert np.isclose(distance, sphere.radius)

    tangent_ray = Ray(origin=np.array([3., -5., 0.]), direction=np.array([0., 1., 0.]))
    expected = np.array([3, 0, 0])
    assert np.all(sphere.find_intersection(tangent_ray) == expected)

    x_epsilon = [0.001, 0, 0]
    tangent_ray.origin += x_epsilon  # check tolerance
    assert np.all(sphere.find_intersection(tangent_ray) == (expected + x_epsilon))


def test_plane_intersection():
    plane = Plane(material=1, normal=np.array([1., 1., 0]), offset=2*np.sqrt(2))

    ray   = Ray(origin=np.array([1., 0., 0.]), direction=np.array([0., 1., 0.])) # intersects
    point = plane.find_intersection(ray)
    assert np.isclose(point, np.array([1., 3., 0.]), rtol=1e-8).all()

    parallel_direction = point - plane.normal * plane.offset
    parallel_ray = Ray(origin=np.array([2., 2., 0]), direction=parallel_direction) # parallel
    assert plane.find_intersection(parallel_ray) == False

    # ray   = Ray(origin=np.array([1., 0., 0.]), direction=np.array([0., -1., 0.])) # behind
    # point = plane.find_intersection(ray)
    # assert point == False


def test_box_intersection():
    box = Box(material=1, center=np.array([3., 3., 0.]), length=2)

    ray = Ray(origin=np.array([1., 0., 0.]), direction=np.array([1., 1., 0.])) # intersects
    point = box.find_intersection(ray)
    assert np.all(point == np.array([3, 2, 0]))

    ray = Ray(origin=np.array([1., 0., 0.]), direction=np.array([2.1, 1., 0.])) # misses
    point = box.find_intersection(ray)
    assert point == False

    ray = Ray(origin=np.array([1., 0., 0.]), direction=np.array([-1., -1., 0.])) # behind
    point = box.find_intersection(ray)
    assert point == False
