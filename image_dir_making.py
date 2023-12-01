# This script is to make an image dir for the database of things

skeleton_og_path = '/home/cilab/teja/sign_langauge_dataset/effective_speaking'

from glob import glob
import os
import cv2

skeleton_vids = glob(os.path.join(skeleton_og_path, '*_skeleton.mp4'))
print(len(skeleton_vids))


output_path = './image_dataset'

from logzero import logger

# for skeleton_vid in skeleton_vids[2:]:

def generate_image_dir(skeleton_vid):

    try:

        # original video
        og_vid_path = skeleton_vid.replace('_skeleton', '')

        og_output_img_dir = os.path.join(output_path, 'output',og_vid_path.split('/')[-1].split('.')[0])

        os.makedirs(og_output_img_dir, exist_ok=True)

        # use ffmpeg to extract all the frames
        # os.system('ffmpeg -i {} {}/%08d.jpg'.format(og_vid_path, og_output_img_dir))

        vid = cv2.VideoCapture(og_vid_path)
        count = 0
        while True:
            ret, frame = vid.read()
            if not ret:
                break

            if count %4 == 0:   
                cv2.imwrite(os.path.join(og_output_img_dir, '{:08d}.jpg'.format(count)), frame)
            count += 1

        # skeleton video
        sk_output_img_dir = os.path.join(output_path, 'input', og_vid_path.split('/')[-1].split('.')[0])    

        os.makedirs(sk_output_img_dir, exist_ok=True)

        # use ffmpeg to extract all the frames
        # os.system('ffmpeg -i {} {}/%08d.jpg'.format(skeleton_vid, sk_output_img_dir))

        vid = cv2.VideoCapture(skeleton_vid)
        count = 0
        while True:
            ret, frame = vid.read()
            if not ret:
                break
            if count %4 == 0:
                cv2.imwrite(os.path.join(sk_output_img_dir, '{:08d}.jpg'.format(count)), frame)
            count += 1


        print('Done with {}'.format(og_vid_path))

    except Exception as e:
        logger.error('Error in {}'.format(skeleton_vid))

import multiprocessing as mp
import time

start = time.time()

# start running things in parallel,
# use all the cores available
pool = mp.Pool(mp.cpu_count())
pool.map(generate_image_dir, skeleton_vids)
pool.close()

end = time.time()
print('Time taken: {}'.format(end - start))

