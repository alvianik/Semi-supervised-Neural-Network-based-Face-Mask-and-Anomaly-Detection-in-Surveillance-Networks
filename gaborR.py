# -*- coding: utf-8 -*-
"""
Created on Fri Apr 16 02:22:54 2021

@author: Dell
"""

import cv2
import glob
from PIL import Image
import os


input_path = glob.glob("E:/Anik Alvi/unsupervised-face-mask-detection/mtcnn-face-detection/code/mtcnn/resizedR/*.jpg") #It is the path to images from where it will read
output_path = "E:/Anik Alvi/unsupervised-face-mask-detection/mtcnn-face-detection/code/mtcnn/gaborR/" ##It is the path to images to where it will write

for file in input_path:
    
    img = cv2.imread(file)
      
    
    theta = 0.0
    sigma = 1
    lamda = 1.5707963267948966
    gamma = 0.5
    
    ksize = 5 #Try 15 for hidden image. Or 9 for others
    phi = 0 #0.8 for hidden image. Otherwise leave it to 0
    kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta, lamda, gamma, phi, ktype=cv2.CV_32F)
    #Now filter the image and add values to a new column
    fimg = cv2.filter2D(img, cv2.CV_8UC3, kernel)  

    im = Image.fromarray(fimg)

    src_fname, ext = os.path.splitext(file)  # split filename and extension
    # construct output filename, basename to remove input directory
    save_fname = os.path.join(output_path, os.path.basename(src_fname)+'.jpg')
    im.save(save_fname)