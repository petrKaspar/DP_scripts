from os.path import join
from PIL import Image
# import Image
from os import listdir
import numpy as np
from os.path import isfile

import cv2


def readMultipleImages(imgDirectoryPath):
    onlyfiles = [f for f in listdir(imgDirectoryPath) if isfile(join(imgDirectoryPath, f))]
    images = np.empty(len(onlyfiles), dtype=object)
    for n in range(0, len(onlyfiles)):
        images[n] = cv2.imread(join(imgDirectoryPath, onlyfiles[n]))
        im1 = Image.open(join(imgDirectoryPath, onlyfiles[n]))
        im2 = im1.rotate(180)
        # brings up the modified image in a viewer, simply saves the image as
        # a bitmap to a temporary file and calls viewer associated with .bmp
        # make certain you have an image viewer associated with this file type
        # im2.show()
        # save the rotated image as d.gif to the working folder
        # you can save in several different image formats, try d.jpg or d.png
        # PIL is pretty powerful stuff and figures it out from the extension
        im2.save(join(imgDirectoryPath, str(n)+onlyfiles[n]))


if __name__ == '__main__':

    imgDirectoryPath = 'images/lego/training4/8x1_yellow/diff'
    readMultipleImages(imgDirectoryPath)
