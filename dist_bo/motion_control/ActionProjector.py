import os
import sys

import numpy as np

tools_root = os.path.join(os.path.dirname(__file__))
sys.path.insert(0, os.path.abspath(tools_root))

from WaypointTracking import BURGER_MAX_LIN_VEL

class ActionProjector(object):
    def __init__(self) -> None:
        self.alpha = 0.5
        