import jax
import jax.numpy as jnp
from jax import grad, jacfwd
import numpy as np


# Create points with coordinates

# Create constraint functions
# A is fixed at 0,0
# AC = 5
# AB = 5
# B is fixed on horizontal axis (y = 0)
# C is fixed on vertical axis (x = 0)

# Build an index map

# Build variables

# Build initial values then initial guess x0

# Loop
# Compute residuals using values
# Derive each residual
# Create Jacobian matrix
# Inverse the Jacobian
# Compute delta x and delta y
