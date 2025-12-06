import jax
import jax.numpy as jnp

from django_project.utils.helpers import (
    vector,
    vector_magnitude,
    cross_product,
    dot_product,
)


def pointandpoint_coincident(point1, point2):
    # point1 must be a jax 1D array: [x, y, 1]
    # point2 must be a jax 1D array: [x, y, 1]
    # Constraint satisfied when function = 0
    v = vector(point1, point2)
    equation = vector_magnitude(v)
    return equation


def coincident_constraint(point1, point2):
    # point1 must be a jax 1D array: [x, y, 1]
    # point2 must be a jax 1D array: [x, y, 1]
    # Constraint satisfied when function = 0
    v = vector(point1, point2)
    function = vector_magnitude(v)
    return function


def distance_constraint(point1, point2, distance):
    # point1 must be a jax 1D array: [x, y, 1]
    # point2 must be a jax 1D array: [x, y, 1]
    # given_distance must be a float
    # Constraint satisfied when residual = 0
    v = vector(point1, point2)
    v_mag_squared = jnp.dot(v, v)  # Faster than taking sqrt
    distance_squared = distance**2
    residual = v_mag_squared - distance_squared
    return residual


def angle_constraint(vector1, vector2, angle):
    # vector1 must be a jax 1D array: [x, y, 1]
    # vector2 must be a jax 1D array: [x, y, 1]
    # angle must be a float [radians]
    # Constraint satisfied when function = 0
    u = dot_product(vector1, vector2)
    v = vector_magnitude(vector1) * vector_magnitude(vector2)
    function = jnp.cos(angle) - u / v
    return function


def parallel_constraint(vector1, vector2):
    # vector1 must be a jax 1D array: [x, y, 1]
    # vector2 must be a jax 1D array: [x, y, 1]
    # Constraint satisfied when function = 0
    function = cross_product(vector1, vector2)
    return function


def perpendicular_constraint(vector1, vector2):
    # vector1 must be a jax 1D array: [x, y, 1]
    # vector2 must be a jax 1D array: [x, y, 1]
    # Constraint satisfied when function = 0
    function = dot_product(vector1, vector2)
    return function


def variable_distance_constraint(point1, point2, minimum, maximum):
    # vector1 must be a jax 1D array: [x, y, 1]
    # vector2 must be a jax 1D array: [x, y, 1]
    # minimum, maximum must be floats [mm]
    # Constraint satisfied when function = 0

    v = vector(point1, point2)
    current_distance = vector_magnitude(v)
    # If current_distance is < minimum
    function = jnp.where(
        current_distance < minimum,
        minimum - current_distance,  # If False, function = minimum - current_distance
        jnp.where(
            current_distance
            > maximum,  # If True, also check if current_distance > maximum,
            current_distance
            - maximum,  # If False, function = current_distance - maximum
            0,
        ),  # If True, function = 0 and constraint is satisfied
    )
    return function


def variable_angle_constraint(vector1, vector2, minimum, maximum):
    # vector1 must be a jax 1D array: [x, y, 1]
    # vector2 must be a jax 1D array: [x, y, 1]
    # minimum, maximum must be floats [radians]
    # Constraint satisfied when function = 0

    u = dot_product(vector1, vector2)
    v = vector_magnitude(vector1) * vector_magnitude(vector2)
    current_angle = jnp.arccos(u / v)

    # If current_angle is < minimum
    function = jnp.where(
        current_angle < minimum,
        minimum - current_angle,  # If False, function = minimum - current_angle
        jnp.where(  # If True, also check if current_angle > maximum,
            current_angle > maximum,
            current_angle - maximum,  # If False, function = current_angle - maximum
            0.0,  # If True, function = 0 and constraint is satisfied
        ),
    )
    return function
