import jax
import jax.numpy as jnp

from .helpers import vector, vector_magnitude, cross_product, dot_product


def constraint_coincident(point1, point2):
    # point1 must be a jax 1D array: [x, y]
    # point2 must be a jax 1D array: [x, y]
    # Constraint satisfied when function = 0
    v = vector(point1, point2)
    function = vector_magnitude(v)
    return function


def constraint_distance(point1, point2, distance):
    # point1 must be a jax 1D array: [x, y]
    # point2 must be a jax 1D array: [x, y]
    # given_distance must be a float
    # Constraint satisfied when function = 0
    v = vector(point1, point2)
    v_mag = vector_magnitude(v)
    function = v_mag - distance
    return function


def constraint_variable_distance(point1, point2, minimum, maximum):
    function = 0
    return function


def constraint_angle(vector1, vector2, angle):
    # vector1 must be a jax 1D array: [x, y]
    # vector2 must be a jax 1D array: [x, y]
    # angle must be a float [radians]
    # Constraint satisfied when function = 0
    u = dot_product(vector1, vector2)
    v = vector_magnitude(vector1) * vector_magnitude(vector2)
    function = jnp.cos(angle) - u / v
    return function


def constraint_variable_angle(vector1, vector2, minimum, maximum):
    function = 0
    return function


def constraint_parallel(vector1, vector2):
    # vector1 must be a jax 1D array: [x, y]
    # vector2 must be a jax 1D array: [x, y]
    # Constraint satisfied when function = 0
    function = cross_product(vector1, vector2)
    return function


def constraint_perpendicular(vector1, vector2):
    # vector1 must be a jax 1D array: [x, y]
    # vector2 must be a jax 1D array: [x, y]
    # Constraint satisfied when function = 0
    function = dot_product(vector1, vector2)
    return function
