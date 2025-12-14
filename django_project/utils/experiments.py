import jax
import numpy as np
import jax.numpy as jnp
from jax import jacfwd

# Lists
# Mutable, ordered
list = [1, 2, 3, 4, 5]

# Tuples
# Immutable, ordered
tuple = (1, 2, 3, 4, 5)

# Dictionaries
# key-value pairs, mutable, fast lookup
dictionary = {
    "name": "Bruce",
    "age": 25,
    "city": "Gotham",
    "hobbies": ["Fighting", "spelunking", "investigating"],
}

# Arrays
# Require numPy or JAX
array = np.array([1, 2, 3, 4, 5])

# Matrices
# Require numPy or JAX
matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

# Experiment 1 - Deriving with a tuple


def constraint(x, y, distance):
    return x**2 + y**2 - distance**2


# residual = constraint(3, 4, 5)
gradient = jacfwd(constraint, argnums=(0, 1))  # Gradient w.r.t x and y only
gradient_values = gradient(3.0, 4.0, 5.0)
print("Gradient values:")
print(f"∂/∂x = {gradient_values[0]}")
print(f"∂/∂y = {gradient_values[1]}")
