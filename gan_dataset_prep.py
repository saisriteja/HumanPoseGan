import cv2
from glob import glob
import os
import random
import numpy as np
from tqdm import tqdm

os.makedirs('./gan_dataset', exist_ok=True)
output_path = '/home/cilab/teja/sign_langauge_dataset/HumanPoser/unq_dataset/output'
# for fol in os.listdir(output_path):

def generate_gan_dataset(fol):
    all_imgs = glob(os.path.join(output_path, fol, '*.jpg'))

    for img in tqdm(all_imgs):
        output_content = cv2.imread(img)
        input_content = cv2.imread(img.replace('output', 'input'))
        input_style = cv2.imread(random.choice(all_imgs))

        output_content = cv2.resize(output_content, (256, 256))
        input_content = cv2.resize(input_content, (256, 256))
        input_style = cv2.resize(input_style, (256, 256))

        # stack the images horizontally
        output_img = np.hstack((input_content, input_style, output_content))
        cv2.imwrite(os.path.join('./gan_dataset', fol  +os.path.basename(img)), output_img)


import multiprocessing as mp
pool = mp.Pool(mp.cpu_count())
pool.map(generate_gan_dataset, os.listdir(output_path))