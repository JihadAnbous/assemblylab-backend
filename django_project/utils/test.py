import jax
import jax.numpy as jnp

from .helpers import vector, vector_magnitude, cross_product, dot_product, transform
from .constraints import constraint_variable_distance, constraint_variable_angle

# Run entire script in terminal with
# `python -m django_project.utils.test`

# Testing vector function
A = jnp.array([0, 1])
B = jnp.array([4, 3])
AB = vector(A, B)
print("Vector: \n", AB)

# Testing vector_magnitude function
a = jnp.array([2, 4])
a_magnitude = vector_magnitude(a)
print("Vector magnitude: \n", a_magnitude)

# Testing cross product function
a = jnp.array([1, 1])
b = jnp.array([2, 2])
cp = cross_product(a, b)
print("Cross product: \n", cp)

# Testing dot product function
c = jnp.array([1, 1])
d = jnp.array([1, 1])
dp = dot_product(c, d)
print("Dot product: \n", dp)

# Testing transform function
square = jnp.array(
    [
        [
            0,
            0,
            1,
            1,
        ],
        [
            0,
            1,
            1,
            0,
        ],
        [1, 1, 1, 1],
    ]
)
angle = jnp.deg2rad(90)
Sx = 1
Sy = 1
transformed_square = transform(square, angle, Sx, Sy)
print("Transformed square: \n", transformed_square)


# Testing variable distance constraint
a = jnp.array([0, 1])
b = jnp.array([1, 2])
c = jnp.array([1, 5])
minimum = 0
maximum = 2
variable_distance_satisfied = constraint_variable_distance(a, b, minimum, maximum)
print("Variable distance satisfied: \n", variable_distance_satisfied)
variable_distance_violated = constraint_variable_distance(a, c, minimum, maximum)
print("Variable distance violated: \n", variable_distance_violated)

# Derivate variable distance constraint
derivative = jax.grad(constraint_variable_distance)
print("Variable distance derivative: \n", derivative)
