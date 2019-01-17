import cv2
import os
import numpy as np

mean_caltech_path = '/home/david/Dropbox/Images/mean_images/mean_caltech.png'
mean_unreal_path = '/home/david/Dropbox/Images/mean_images/mean_unreal.png'
mean_citymapped23_path = '/home/david/Dropbox/Images/mean_images/mean_citymapped23.png'
mean_citymapped37_path = '/home/david/Dropbox/Images/mean_images/mean_citymapped37.png'
mean_mixed_path = '/home/david/Dropbox/Images/mean_images/mean_mixed.png'

img_caltech = cv2.imread(mean_caltech_path)
cv2.imshow(img_caltech)
cv2.waitKey()