from __future__ import print_function

import numpy as np
import cv2
import os, math
from scipy.optimize import leastsq
from PIL import Image

TOP_Y_MIN = -30
TOP_Y_MAX = +30
TOP_X_MIN = 0
TOP_X_MAX = 100
TOP_Z_MIN = -3.5
TOP_Z_MAX = 0.6

TOP_X_DIVISION = 0.2
TOP_Y_DIVISION = 0.2
TOP_Z_DIVISION = 0.3

cbox = np.array([[0, 70.4], [-40, 40], [-3, 2]])

print("hello world")

def load_image(img_filename):
    return cv2.imread(img_filename)