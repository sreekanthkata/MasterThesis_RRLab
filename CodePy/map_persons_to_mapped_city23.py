import cv2
import os
import xml.etree.ElementTree as ET
import glob
from cyclegan1 import main2 as cycgan
import tensorflow as tf
import numpy as np

#root_dir = '/home/d_sperber/tfdata/unreal_60m/UnrealDataset/Mapped480'
unreal_image_dir = '/home/d_sperber/tfdata/unreal_60m/UnrealDataset/JPEGImages480' #unreal bilder mit gemappten personen
mask_dir = '/home/d_sperber/tfdata/unreal_60m/UnrealDataset/Seg480/'
image_dir = '/home/d_sperber/tfdata/unreal_60m/UnrealDataset/Mapped480/JPEGImages_city23' # gemappte city
destination_dir = '/home/d_sperber/tfdata/unreal_60m/UnrealDataset/Mapped480/JPEGImages_cityandorigpersons23'

def main():
    
    counter = 0
    for filename in os.listdir(image_dir):
        # load images
        unreal_image = cv2.imread(os.path.join(unreal_image_dir, filename))
        #unreal_image = cv2.resize(unreal_image, (640,480))
        img = cv2.imread(os.path.join(image_dir, filename))
        mask = cv2.imread(os.path.join(mask_dir, filename.replace('visible', 'class')))
        
        new_image = replace_pers(img, unreal_image, mask)
        
        #cv2.imshow('img', new_image)
        #cv2.waitKey()
        cv2.imwrite(os.path.join(destination_dir, filename), new_image)
        
        counter += 1
        print('Saved', counter, 'images ...')
        
def replace_pers(image, unreal_image, mask):
    mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
    new_image = image
    
    for h in range(new_image.shape[0]):
        for w in range(new_image.shape[1]):
            if(mask[h][w] > 70):
                new_image[h][w][0] = unreal_image[h][w][0]
                new_image[h][w][1] = unreal_image[h][w][1]
                new_image[h][w][2] = unreal_image[h][w][2]    
    return new_image
    
            
if __name__ == '__main__':
    main()            
            