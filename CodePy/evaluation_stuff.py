import numpy as np
from matplotlib import pyplot as plt
from xml.etree import cElementTree as ET
import sys
import os
from seaborn import heatmap
import cv2
import tensorflow as tf
from object_detection.data_decoders.tf_example_decoder import TfExampleDecoder

def get_heatmap_from_xml(directory):
    
    annotations = os.listdir(directory)
    hmap = np.zeros((1200, 1600))
    counter = 0
    
    f, ax = plt.subplots(figsize=(9, 6))
     
    for a in annotations:
        
        path = os.path.join(directory, a)
        tree = ET.parse(path)
        root = tree.getroot()
      #  print(path)
        for obj in root.findall("object"):
            if obj is not None:
                minX = int(obj.find("bndbox").find("xmin").text)
                minY =  int(obj.find("bndbox").find("ymin").text)
                maxX =  int(obj.find("bndbox").find("xmax").text)
                maxY =  int(obj.find("bndbox").find("ymax").text)
                
             #  print(minY, maxY)
              #  print(hmap.shape)
              #  print( hmap[minY:maxY, minX:maxX].shape)
              #  print((maxY-minY, maxX-minX))
                hmap[minY:maxY, minX:maxX] += 1
        counter += len(root.findall("object"))
        
    hmap /= counter
    plt.title("Heatmap of bounding box distribution in Unreal Dataset")
    heatmap(hmap, annot=False, xticklabels=False, yticklabels=False, ax=ax)
    plt.show()
    
def get_height_histo_from_xml(directory):
    
    annotations = os.listdir(directory)
    f, ax = plt.subplots(figsize=(9, 6))
    heights = []

    for a in annotations:

        path = os.path.join(directory, a)
        tree = ET.parse(path)
        root = tree.getroot()
        
        for obj in root.findall("object"):
            if obj is not None:
                height = int(obj.find("bndbox").find("ymax").text) - int(obj.find("bndbox").find("ymin").text)
                heights.append(height)
    
    plt.xlabel("heights in pixels")
    plt.ylabel("number of boxes ")
    plt.title("Distribution of bounding box heights")
    ax.hist(heights, 120, rwidth=0.9)
    plt.show()
    
def get_number_histo_from_xml(directory):
    
    annotations = os.listdir(directory)
    f, ax = plt.subplots(figsize=(9, 6))
    noms = []
    
    for a in annotations:
        
        path = os.path.join(directory, a)
        tree = ET.parse(path)
        root = tree.getroot()
        
        noms.append(len(root.findall("object")))
    
    plt.xlabel("number of tagged objects")
    plt.ylabel("number of data points")
    plt.title("Distibution of tagged objects per Image in Unreal Dataset")
    ax.hist(noms, bins=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15], rwidth=0.9, align="mid")
    
    print(len(noms) - np.count_nonzero(noms))
    plt.show()
    
def get_avarage_bbox_ratio(directory):
    
    annotations = os.listdir(directory)
    f, ax = plt.subplots(figsize=(9, 6))
    ratios = []
    
    for a in annotations:

        path = os.path.join(directory, a)
        tree = ET.parse(path)
        root = tree.getroot()
        
        for obj in root.findall("object"):
            if obj is not None:
                ratio = (int(obj.find("bndbox").find("xmax").text) - int(obj.find("bndbox").find("xmin").text)) / (int(obj.find("bndbox").find("ymax").text) - int(obj.find("bndbox").find("ymin").text) )
            
   
                ratios.append(ratio)
        
    return np.mean(ratios)
    
def get_imgsize_pctg_histo_from_xml(directory):
    
    annotations = os.listdir(directory)
    f, ax = plt.subplots(figsize=(9, 6))
    pctgs = []
    
    for a in annotations:

        path = os.path.join(directory, a)
        tree = ET.parse(path)
        root = tree.getroot()
        
        for obj in root.findall("object"):
            if obj is not None:
                pctg = (int(obj.find("bndbox").find("ymax").text) - int(obj.find("bndbox").find("ymin").text) )*(int(obj.find("bndbox").find("xmax").text) - int(obj.find("bndbox").find("xmin").text))
                pctg /= 1600*1200
                pctg *= 100
   
                pctgs.append(pctg)

    plt.xlabel("Percentage of image size")
    plt.ylabel("Fraction of bounding boxes")
    plt.title("Distribution of bounding box sizes as percentage of image in Unreal Dataset")
    ax.hist(pctgs, 100, density=True, rwidth=0.9)
    plt.show()
    
def calculate_mean_image(directory):
        
    mean = np.zeros((1200, 1600, 3))
    
    images_paths = os.listdir(directory)
    counter = 0
    
    for img_path in images_paths:
        
        img = cv2.imread(os.path.join(directory, img_path), cv2.IMREAD_COLOR)
        mean += img
        print(counter, len(images_paths))
        counter += 1
        
    mean /= counter
    cv2.imwrite("/home/d_sperber/tfdata/unreal_60m/UnrealDataset/Mapped480/unreal_mean_img.png", mean)

def calculate_img_mean_from_record(record):
    
    mean = np.zeros((1200, 1600, 3))
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(record)
    
    features = tf.parse_single_example(serialized_example,
  # Defaults are not specified since both keys are required.
    features={
      'image_raw': tf.FixedLenFeature([], tf.string),
      'label': tf.FixedLenFeature([], tf.int64),
      'height': tf.FixedLenFeature([], tf.int64),
      'width': tf.FixedLenFeature([], tf.int64),
      'depth': tf.FixedLenFeature([], tf.int64)
    })
 
def read_and_decode(filename_queue):
    
    decoder = TfExampleDecoder()
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    
    tensorDict = decoder.decode(serialized_example)
  
    image =tensorDict["image"]
    boxes = tensorDict["groundtruth_boxes"]
    
    #depth = tf.cast(features['depth'], tf.int32)
    
    return image, boxes


def get_mean_from_record(FILE):
    
    with tf.Session() as sess:
        
        
        mean = np.zeros((480, 640, 3))
        
        filename_queue = tf.train.string_input_producer([ FILE ])
        image, _ = read_and_decode(filename_queue)
      
        #image = tf.reshape(image, [height, width, 3])
        image.set_shape([480, 640, 3])
        init_op = tf.initialize_all_variables()
        sess.run(init_op)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        
        for i in range(25093):
            frame = sess.run([image])
        
            img = cv2.cvtColor(frame[0], cv2.COLOR_RGB2BGR)
            #img.save( "output/" + str(i) + '-train.png')
            mean += img

           
        coord.request_stop()
        coord.join(threads)
        mean /= 25093
        cv2.imwrite("/home/jakub/mean_caltech.png", mean)
        
def get_heatmap_from_record(FILE):
    
    with tf.Session() as sess:
        
        f, ax = plt.subplots(figsize=(9, 6))
        
        hmap = np.zeros((480, 640))
        
        filename_queue = tf.train.string_input_producer([ FILE ])
        _, boxes = read_and_decode(filename_queue)
      
        #image = tf.reshape(image, [height, width, 3])
       
        init_op = tf.initialize_all_variables()
        sess.run(init_op)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        counter = 0
        
        for i in range(25093):
            bbs = sess.run([boxes])
            for bb in bbs[0]:
                hmap[int(480 * bb[0]): int(480*bb[2]), int(640*bb[1]): int(640*bb[3])] += 1
                counter += 1
           
        coord.request_stop()
        coord.join(threads)
        hmap  /= counter
        
        plt.title("Heatmap of bounding box distribution in Caltech dataset")
        heatmap(hmap, annot=False, xticklabels=False, yticklabels=False, ax=ax)
        plt.show()

    
    
    
def main():
    
  #  get_heatmap_from_xml("/mnt/bigtmp/datasets/rrlab/unreal/Pawlak/UnrealDataset/Annotations/")
   # get_height_histo_from_xml("/mnt/bigtmp/datasets/rrlab/unreal/Pawlak/UnrealDataset/Annotations/")
   # get_imgsize_pctg_histo_from_xml("/mnt/bigtmp/datasets/rrlab/unreal/Pawlak/UnrealDataset/Annotations/")
    calculate_mean_image(sys.argv[1])
    #get_mean_from_record(sys.argv[1])
    #get_heatmap_from_record("/mnt/public_work/temp/d_sperber/caltech_tf/caltech_train_person_30.record")
    #print(get_avarage_bbox_ratio("/mnt/bigtmp/datasets/rrlab/unreal/Pawlak/UnrealDataset/Annotations/"))
    #get_number_histo_from_xml("/mnt/bigtmp/datasets/rrlab/unreal/Pawlak/UnrealDataset/Annotations/")
    
    
main()
    
    
    
