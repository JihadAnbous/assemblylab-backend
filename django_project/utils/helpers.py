import jax
import jax.numpy as jnp


def point(point):
    # point.x must be a float
    # point.y must be a float
    # Returns a jax 1D array: [x, y]
    x = point.x
    y = point.y
    result = jnp.array([x, y])
    return result


def vector(point1, point2):
    # point1 must be a jax 1D array: [x, y]
    # point2 must be a jax 1D array: [x, y]
    # Returns a jax 1D array: [x, y]
    x1 = point1[0]
    y1 = point1[1]
    x2 = point2[0]
    y2 = point2[1]
    result = jnp.array([x2 - x1, y2 - y1])
    return result


def vector_magnitude(vector):
    # vector must be a jax 1D array: [x, y]
    # Returns the magnitude of a vector
    x = vector[0]
    y = vector[1]
    result = jnp.sqrt(x**2 + y**2)
    return result


def cross_product(vector1, vector2):
    # vector1 must be a jax 1D array: [x, y]
    # vector2 must be a jax 1D array: [x, y]
    # Returns the cross product of 2 2D vectors
    result = jnp.cross(vector1, vector2)
    return result


def dot_product(vector1, vector2):
    # vector1 must be a jax 1D array: [x, y]
    # vector2 must be a jax 1D array: [x, y]
    # Returns the dot product of 2 2D vectors
    result = jnp.dot(vector1, vector2)
    return result


# Transform a jax array
def transform(a, theta, Sx, Sy):
    # Throw an error if the input matrix 'a' is not a numpy array
    if not isinstance(a, np.ndarray):
        raise ValueError("Input 'a' must be a numpy array.")

    # Define the transformation matrix t (rotation + translation)
    t = np.array(
        [[cosd(theta), -sind(theta), Sx], [sind(theta), cosd(theta), Sy], [0, 0, 1]]
    )
    # Matrix multiplication of t and a (assuming a is a numpy array)
    b = np.dot(t, a)

    return b
