import jax
import jax.numpy as jnp


def point(point):
    # point.x must be a float
    # point.y must be a float
    # Returns a jax 1D array: [x, y, 1]
    x = point.x
    y = point.y
    result = jnp.array([x, y, 1])
    return result


def vector(point1, point2):
    # point1 must be a jax 1D array: [x, y, 1]
    # point2 must be a jax 1D array: [x, y, 1]
    # Returns a jax 1D array: [x, y, 1]
    x1 = point1[0]
    y1 = point1[1]
    x2 = point2[0]
    y2 = point2[1]
    result = jnp.array([x2 - x1, y2 - y1, 1])
    return result


def vector_magnitude(vector):
    # vector must be a jax 1D array: [x, y, 1]
    # Returns the magnitude of a vector
    x = vector[0]
    y = vector[1]
    result = jnp.sqrt(x**2 + y**2)
    return result


def cross_product(vector1, vector2):
    # vector1 must be a jax 1D array: [x, y, 1]
    # vector2 must be a jax 1D array: [x, y, 1]
    # Returns the cross product of 2 2D vectors
    result = jnp.cross(vector1, vector2)
    return result


def dot_product(vector1, vector2):
    # vector1 must be a jax 1D array: [x, y, 1]
    # vector2 must be a jax 1D array: [x, y, 1]
    # Returns the dot product of 2 2D vectors
    result = jnp.dot(vector1, vector2)
    return result


def transform(matrix, angle, Sx, Sy):
    # matrix must be a matrix formed of jax 1D arays: [x, y, 1]
    # angle must be the angle of rotation in radians
    # Sx and Sy are the linear transformation values in the x and y directions
    # The resultant is the transformed matrix formed of jax 1D arays: [x, y, 1]

    t = jnp.array(
        [
            [jnp.cos(angle), -jnp.sin(angle), Sx],
            [jnp.sin(angle), jnp.cos(angle), Sy],
            [0, 0, 1],
        ]
    )
    result = jnp.round(jnp.dot(t, matrix), decimals=6)
    return result
