import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
imported = Axes3D

from skspatial.objects import Point, Line, Vector, Sphere as SkSphere


def new_figure():
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    return ax

def plot_ray(ray, ax, color='g'):
    line = Line(point=ray.origin, direction=ray.direction)
    line.plot_3d(ax, t_1=10, c=color)

def plot_box(box, ax, color='r'):
    min_point = box._box_min
    max_point = box._box_max
    for i in range(3):
        new_point = min_point.copy()
        new_point[i] = max_point[i]
        Line.from_points(min_point, new_point).plot_3d(ax, c=color)

    for i in range(3):
        new_point = max_point.copy()
        new_point[i] = min_point[i]
        Line.from_points(max_point, new_point).plot_3d(ax, c=color)

def plot_sphere(sphere, ax, color='b'):
    sphere = SkSphere(sphere.center, sphere.radius)
    sphere.plot_3d(ax, alpha=0.2, c=color)

def plot_point(point, ax, color='k'):
    Point(point).plot_3d(ax, c=color)
