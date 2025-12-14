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
# Degrees to radians
a = jnp.deg2rad(45)
print(f"45 degrees is", a, "radians")


# Derivative or gradient - example 1
def random_equation(time):
    YIELD_KT = 15.0
    BURST_HEIGHT = 1000.0

    return YIELD_KT * time**2 + 3 * time + BURST_HEIGHT


random_equation_gradient = jax.grad(random_equation)
print(f"Random equation gradient:\n", random_equation_gradient)


# Derivative or gradient - example 1
def speed_equation(time, speed):
    YIELD_KT = 15.0
    BURST_HEIGHT = 1000.0

    return YIELD_KT * time**2 + 3 * time + BURST_HEIGHT * speed


speed_equation_gradient = jax.grad(speed_equation)
print(f"Speed equation gradient:\n", speed_equation_gradient)


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
