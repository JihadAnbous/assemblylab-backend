import jax
import jax.numpy as jnp


# Function
def speed_equation(time, speed):
    return 15 * time**2 + 3 * time + 1000 * speed


# Gradient
grad_fn = jax.grad(speed_equation)

print("Gradient at (time=2.0, speed=5.0):")
print(grad_fn((2.0, 5.0)))
