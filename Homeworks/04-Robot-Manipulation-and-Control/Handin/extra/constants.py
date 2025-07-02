import numpy as np

# we scale the depth value so that it can fit in np.uint16 and use image compression algorithms
DEPTH_IMG_SCALE = 16384

# simulation initialization settings
TABLE_HEIGHT = 0.5
OBJ_INIT_TRANS = np.array([0.535, 0.36, 0.83])
OBJ_RAND_RANGE = 0.3
OBJ_RAND_SCALE = 0.05

# clip the point cloud to a box
# Updated based on Assignment2 actual data analysis from 10 samples (9,216,000 points)
# Using 5%-95% percentiles with 2cm margin for robustness
PC_MIN = np.array([0.2, 0, 0.640])
PC_MAX = np.array([1, 0.8, 0.85])
