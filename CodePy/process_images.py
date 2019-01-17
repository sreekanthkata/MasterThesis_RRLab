import cv2
import os
import numpy as np
import random


unreal_person_path = '/home/d_sperber/tfdata/unreal_60m/UnrealDataset/Mapped480/unreal_with_persons1200/'
unreal_path = '/home/d_sperber/tfdata/unreal_60m/UnrealDataset/JPEGImages'
mapped_city23_path = '/home/d_sperber/tfdata/unreal_60m/UnrealDataset/Mapped480/JPEGImages_city23/'
mapped_city37_path = '/home/d_sperber/tfdata/unreal_60m/UnrealDataset/Mapped480/JPEGImages_city37/'
mask_path = '/home/d_sperber/tfdata/unreal_60m/UnrealDataset/Seg480/'

unreal_mean = '/home/d_sperber/tfdata/unreal_60m/UnrealDataset/Mapped480/mean_unrealwithpersons.png'
mapped_mean23 = '/home/d_sperber/tfdata/unreal_60m/UnrealDataset/Mapped480/mean_citymapped23.png'
mapped_mean37 = '/home/d_sperber/tfdata/unreal_60m/UnrealDataset/Mapped480/mean_citymapped37.png'

out_dir1 = '/home/d_sperber/Dropbox/Experiments/Example_images_mixed'
#out_dir2 = '/home/d_sperber/tfdata/unreal_60m/UnrealDataset/Mapped480/JPEGImages'

def clear_humans(orig_img, img, mask, percentage=1.0):
    mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
    #testing cooment
    
    for h in range(img.shape[0]):
        for w in range(img.shape[1]):
            if(mask[h][w] > 70):
                img[h][w][0] = min(int(float(img[h][w][0])*percentage) + int(float(orig_img[h][w][0]))*(1.0-percentage), 255)
                img[h][w][1] = min(int(float(img[h][w][1])*percentage) + int(float(orig_img[h][w][1]))*(1.0-percentage), 255)
                img[h][w][2] = min(int(float(img[h][w][2])*percentage) + int(float(orig_img[h][w][2]))*(1.0-percentage), 255)
    
    

def process():
    
    mean_unreal = cv2.resize(cv2.imread(unreal_mean), (640,480))
    mean_mapped23 = cv2.imread(mapped_mean23)
    mean_mapped37 = cv2.imread(mapped_mean37)

    counter = 0
    for filename in os.listdir('/home/d_sperber/tfdata/unreal_60m/UnrealDataset/Mapped480/JPEGImages_city23/'):
        
        unreal_person = cv2.resize(cv2.imread(os.path.join(unreal_person_path, filename)), (640,480))
        unreal = cv2.resize(cv2.imread(os.path.join(unreal_path, filename)), (640,480))
        mapped23 = cv2.imread(os.path.join(mapped_city23_path, filename))
        mapped37 = cv2.imread(os.path.join(mapped_city37_path, filename))
        mask_name = filename.replace('visible', 'class')
        mask = cv2.imread(os.path.join(mask_path, mask_name))
        
        unreal_blurr = cv2.GaussianBlur(unreal,(21,21),0)
        
        # mapped23 + unreal_person 
        result1 = cv2.addWeighted(unreal_blurr, 0.6, mean_mapped23, 1, 1)
        result1 = cv2.addWeighted(result1, 0.4, mapped23, 0.5, 1)
        clear_humans(unreal, result1, mask, 0.3)        
        # mapped37 + unreal_person 
        result2 = cv2.addWeighted(unreal_blurr, 0.6, mean_mapped37, 1, 1)
        result2 = cv2.addWeighted(result2, 0.4, mapped37, 0.5, 1)
        clear_humans(unreal, result2, mask, 0.3)        
       
        cv2.imshow("u", unreal)
        cv2.imshow("1", result1)
        cv2.imshow("2", result2)
        
        k = cv2.waitKey()
        if k == 10:
            print 'save ...'
            cv2.imwrite(os.path.join(out_dir1, 'img_' + str(counter) + '_mixed23.jpg'), result1)
            cv2.imwrite(os.path.join(out_dir1, 'img_' + str(counter) + '_mixed37.jpg'), result2)
            cv2.imwrite(os.path.join(out_dir1, 'img_' + str(counter) + '_unreal.jpg'), unreal)
            counter += 1
        else:
            print 'skip ...'
 
        """
        # mapped23 + mapped_person 
        result3 = cv2.addWeighted(unreal, 0.7, mean_mapped23, 0.7, 1)
        result3 = cv2.addWeighted(result3, 0.4, mapped23, 0.5, 1)
        clear_humans(unreal_person, result3, mask)        
        # mapped37 + mapped_person 
        result4 = cv2.addWeighted(unreal, 0.7, mean_mapped37, 0.7, 1)
        result4 = cv2.addWeighted(result4, 0.4, mapped37, 0.5, 1)
        clear_humans(unreal_person, result4, mask)        
        """
        # save files
        """
        cv2.imwrite(os.path.join(out_dir1, filename.replace('.jpg', '_1.jpg')), result1)
        cv2.imwrite(os.path.join(out_dir1, filename.replace('.jpg', '_2.jpg')), result2)
        cv2.imwrite(os.path.join(out_dir2, filename.replace('.jpg', '_1.jpg')), result1)
        cv2.imwrite(os.path.join(out_dir2, filename.replace('.jpg', '_2.jpg')), result2)
        """
        #cv2.imwrite(os.path.join(out_dir2, filename.replace('.jpg', '_1.jpg')), result3)
        #cv2.imwrite(os.path.join(out_dir2, filename.replace('.jpg', '_2.jpg')), result4)
        
        print('save images ' + str(counter) + '/' + str(len(os.listdir('/home/d_sperber/tfdata/unreal_60m/UnrealDataset/Mapped480/JPEGImages_city23/'))))
        
        
        #cv2.imshow('result1', result1)
        #cv2.imshow('result2', result2)
        #cv2.imshow('result3', result3)
        #cv2.imshow('result4', result4)
        #cv2.waitKey()

def main():
    process()

if __name__ == '__main__':
   main()