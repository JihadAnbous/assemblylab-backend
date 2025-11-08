import jax
import jax.numpy as jnp

from .helpers import vector, vector_magnitude, cross_product, dot_product

# Run entire script in terminal with
# `python -m django_project.utils.test`

# Testing vector function
A = jnp.array([[0], [1]])
B = jnp.array([[4], [3]])
AB = vector(A, B)
print("Vector: \n", AB)

# Testing vector_magnitude function
a = jnp.array([[2], [4]])
a_magnitude = vector_magnitude(a)
print("Vector magnitude: \n", a_magnitude)

# Testing cross product function
a = jnp.array([[1], [1]])
b = jnp.array([[2], [2]])
cp = cross_product(a, b)
print("Cross product: \n", cp)

# Testing dot product function
a = jnp.array([[0], [1]])
b = jnp.array([[1], [0]])
dp = dot_product(a, b)
print("Dot product: \n", dp)
