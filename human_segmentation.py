import cv2
import math
import numpy as np
import mediapipe as mp
import os

mp_selfie_segmentation = mp.solutions.selfie_segmentation
BG_COLOR = (0,0,0) # black
MASK_COLOR = (255, 255, 255) # white

def get_segemented_person(path, MASK_COLOR, selfie_segmentation):
    img  = cv2.imread(path)
    results = selfie_segmentation.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    # Generate solid color images for showing the output selfie segmentation mask.
    mask = np.zeros(img.shape, dtype=np.uint8)
    mask[:] = MASK_COLOR

    condition = np.stack((results.segmentation_mask,) * 3, axis=-1) > 0.1
    
    # white the background and leave the person
    output_image = np.where(condition, img, mask)
    
    op_path = path.replace('cropped_images', 'cropped_images_segmented')
    os.makedirs(os.path.dirname(op_path), exist_ok=True)

    cv2.imwrite(op_path, output_image)

def segment_database_persons(path):
    with mp_selfie_segmentation.SelfieSegmentation() as selfie_segmentation:
        get_segemented_person(path, MASK_COLOR, selfie_segmentation)



from glob import glob
# path = '/home/cilab/teja/sign_langauge_dataset/HumanPoser/cropped_images/output/1176624_1a1/00000008.jpg'

paths = glob('/home/cilab/teja/sign_langauge_dataset/HumanPoser/cropped_images/output/*/*.jpg')

from multiprocessing import Pool
import multiprocessing as mp

from logzero import logger

logger.info('starting the process')

with Pool(mp.cpu_count()) as p:
    p.map(segment_database_persons, paths)

logger.info('done')
# segment_database_persons(paths[0])

os.system('cp -r cropped_images/input/ cropped_images_segmented/')