import pytest
from classes import Camera, Set, Material, Light, Sphere, Plane, Box, Ray

import numpy as np

def test_intersection():
    ray = Ray(origin=np.array([0., 3., 0.]), direction=np.array([0.73723662, -0.67311481, 0.05829765]))
    sphere = Sphere(material=3, center=np.array([2., 0., 0.]), radius=1.0)

    point = sphere.find_intersection(ray)

    distance = np.linalg.norm(point - sphere.center)

    assert np.isclose(distance,sphere.radius)