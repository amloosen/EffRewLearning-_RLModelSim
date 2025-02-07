import numpy as np

def create_param_grid(lb, ub, stepsPerDim):
    """
    Create a parameter grid for a given set of lower and upper bounds and steps per dimension.
    
    Parameters
    ----------
    lb : list or array of floats
        Lower bounds for each parameter dimension.
    ub : list or array of floats
        Upper bounds for each parameter dimension.
    stepsPerDim : list or array of ints
        Number of steps (interval points) for each dimension.

    Returns
    -------
    param_grid : numpy.ndarray
        A 2D array of shape (N, p), where N = product of stepsPerDim,
        p = number of parameters. Each row is one combination of parameter values.
    """

    # p = number of parameters
    p = len(lb)
    # Create the 1D arrays for each dimension
    tmp_grids = []
    for i in range(p):
        tmp_array = np.linspace(lb[i], ub[i], stepsPerDim[i])
        tmp_grids.append(tmp_array)

    # Use numpy.meshgrid to create an n-dimensional grid
    # indexing='ij' ensures we mimic ndgrid
    mg = np.meshgrid(*tmp_grids, indexing='ij')

    # mg is a list of p arrays, each shape = stepsPerDim (depending on dimension)
    # Flatten each dimension
    flattened = [m.ravel() for m in mg] 
    # shape after flatten: each 'm' is array of size N=product(stepsPerDim)

    # Stack them along axis=1 => shape (N, p)
    param_grid = np.stack(flattened, axis=-1)
    return param_grid.astype(np.float32)  # or keep float64 if you prefer
