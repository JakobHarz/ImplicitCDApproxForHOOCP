# numpy print options
import numpy as np
np.set_printoptions(precision=4)

# ipopt default options
IPOPT_OPTIONS_LINEAR_SOLVER: str = 'MA27'
IPOPT_OPTIONS_MAX_ITER: int = 2000
IPOPT_OPTIONS_MU_TARGET_HOMOTOPY: float = 1e-4


# logger
import logging as lg
lg.basicConfig(level=lg.INFO,
                    format='[%(asctime)s-%(levelname)s][%(module)s] %(message)s',
                    datefmt='%H:%M:%S')
logging = lg.getLogger("DefaultLogger")


# Plotting Formatting Options
PLT_GRID_ALPHA: float = 0.4