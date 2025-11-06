import jax
import jax.numpy as jnp


def vector(point1, point2):
    # point1 must be a 1x2 jax array [mm]
    # point2 must be a 1x2 jax array [mm]
    x1 = point1[0, 0]
    y1 = point1[1, 0]
    x2 = point2[0, 0]
    y2 = point2[1, 0]
    vector = jnp.array([[x2 - x1], [y2 - y1]])
    return vector


# Testing vector
A = jnp.array([[0], [1]])
B = jnp.array([[4], [3]])
AB = vector(A, B)
print(AB)
