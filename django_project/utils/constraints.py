import jax
import jax.numpy as jnp


def constraint_distance(given_point, constrained_point, given_distance):
    # given_point must be a 1x2 jax array [mm]
    # constrained_point must be a 1x2 jax array [mm]
    # given_distance must be a float [mm]
    x1 = given_point[0]
    y1 = given_point[1]
    x = constrained_point[0]
    y = constrained_point[1]
    function = jnp.sqrt((x - x1) ** 2 + (y - y1) ** 2) - given_distance
    return function


def constraint_angle(vector1, vector2, given_angle):
    # vector1 must be a 1x2 jax array [mm]
    # vector2 must be a 1x2 jax array [mm]
    # given_angle must be a float [radians]
    return None
