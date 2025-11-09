import jax
import jax.numpy as jnp


arr = jnp.array([[1.0, 2.0], [3.0, 4.0]])
brr = jnp.array([[3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
crr = jnp.array([[1.0, 2.0], [3.0, 4.0]])

# Change entire 1st row to 99
changed_row = arr.at[0].set(99.0)
print(f"Changed row:\n", changed_row)
# Change element in 1st row and last column
changed_element = brr.at[0, -1].set(99.0)
print(f"Changed element:\n", changed_element)
# Element-wise addition
drr = arr + crr
print(f"drr:\n", drr)
# Element-wise multiplication
err = arr * crr
print(f"err:\n", err)
# Dot product
frr = jnp.dot(arr, crr)
print(f"frr:\n", frr)


# Derivative or gradient - example 1
def random_equation(time):
    YIELD_KT = 15.0
    BURST_HEIGHT = 1000.0

    return YIELD_KT * time**2 + 3 * time + BURST_HEIGHT


random_equation_gradient = jax.grad(random_equation)
print(f"Random equation gradient:\n", random_equation_gradient)


# Derivative or gradient - example 2
def constraint_distance(x_1, y_1, d_1):
    return


def constraint_distance(given_point, constrained_point, given_distance):
    # given_point must be a 1x2 jax array
    # constrained_point must be a 1x2 jax array
    # given_distance must be a float
    x1 = given_point[0]
    y1 = given_point[1]
    x = constrained_point[0]
    y = constrained_point[1]
    function = jnp.sqrt((x - x1) ** 2 + (y - y1) ** 2) - given_distance
    return function


# Example 1: Points that satisfy the constraint
a = jnp.array([0.0, 0.0])
b = jnp.array([3.0, 4.0])  # This is distance 5 from origin
d = 5.0

result = constraint_distance(a, b, d)
print(f"Constraint value: {result}")  # Should be 0.0 (constraint satisfied)

# Derive the function
constraint_distance_gradient = jax.grad(constraint_distance)
print(constraint_distance_gradient)

current_angle = 45
minimum = 0
maximum = 90
# Example:
jnp.where(
    current_angle < minimum,
    minimum - current_angle,  # returned if condition is True
    0.0,  # returned if condition is False
)

current_angle_cos = 25
# Example:
jnp.clip(current_angle_cos, -1.0, 1.0)

# If current_angle_cos = 1.05  → returns 1.0 (clamped to max)
# If current_angle_cos = 0.5   → returns 0.5 (unchanged)
# If current_angle_cos = -1.2  → returns -1.0 (clamped to min)
