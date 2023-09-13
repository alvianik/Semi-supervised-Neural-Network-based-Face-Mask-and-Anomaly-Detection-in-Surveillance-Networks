# -*- coding: utf-8 -*-
"""
Created on Fri Apr 16 03:27:39 2021

@author: Dell
"""

import cv2
import glob
from PIL import Image
import os

input_path = glob.glob("E:/Anik Alvi/unsupervised-face-mask-detection/mtcnn-face-detection/code/mtcnn/resized2/*.jpg") #It is the path to images from where it will read
output_path = "E:/Anik Alvi/unsupervised-face-mask-detection/mtcnn-face-detection/code/mtcnn/laplacian2/" ##It is the path to images to where it will write

for file in input_path:
    
    img = cv2.imread(file)
    
    laplacian = cv2.Laplacian(img,cv2.CV_8UC3)

    im = Image.fromarray(laplacian)


    src_fname, ext = os.path.splitext(file)  # split filename and extension
    # construct output filename, basename to remove input directory
    save_fname = os.path.join(output_path, os.path.basename(src_fname)+'.jpg')
    im.save(save_fname)