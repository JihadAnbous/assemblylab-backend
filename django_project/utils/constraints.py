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
    # vector1 must be a jax 1D array: [x, y]
    # vector2 must be a jax 1D array: [x, y]
    # minimum, maximum must be floats [radians]
    # Constraint satisfied when function = 0

    u = dot_product(vector1, vector2)
    v = vector_magnitude(vector1) * vector_magnitude(vector2)
    current_angle_cos = u / v

    # Convert to actual angle for comparison
    current_angle = jnp.arccos(jnp.clip(current_angle_cos, -1.0, 1.0))

    # Return 0 if within bounds, otherwise return violation amount
    # This creates a penalty that becomes 0 when constraint is satisfied
    function = jnp.where(
        current_angle < minimum,
        minimum - current_angle,  # Violation below minimum
        jnp.where(
            current_angle > maximum,
            current_angle - maximum,  # Violation above maximum
            0.0,  # Within bounds
        ),
    )
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
