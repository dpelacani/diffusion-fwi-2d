import numpy as np

VMIN            = 1480.0   # minimum physical velocity (m/s)
VMAX            = 3000.0   # maximum physical velocity (m/s)
WATER_VELOCITY  = 1480.0   # background water velocity (m/s)
SKULL_THRESH_MS = 1650.0   # threshold value separating soft tissue from skull (m/s)

# Normalized VMIN under AcousticNorm + Normalize(0.5, 0.5)
# = (log(1480/3000)+1 - 0.5) / 0.5 = -0.4172
WATER_FILL = float((np.log(VMIN / 3000.0) + 1.0 - 0.5) / 0.5)
NORM_MIN = WATER_FILL
NORM_MAX = float((np.log(VMAX / 3000.0) + 1.0 - 0.5) / 0.5)