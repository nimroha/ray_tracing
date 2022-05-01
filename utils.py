import numpy as np
import cv2 as cv

def write_img(img, img_path):
    img = (img * 255).astype(np.uint8)
    bgr = cv.cvtColor(img, cv.COLOR_RGB2BGR)
    cv.imwrite(img_path, bgr) # TODO how to save?
