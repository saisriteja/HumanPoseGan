import torch
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
import cv2
from glob import glob
# imgs = glob("/home/cilab/teja/sign_langauge_dataset/HumanPoser/image_dataset/output/*/*.jpg")
import random
import os

imgs = []
path = './image_dataset/output'
for fol in os.listdir(path):
    all_i = glob(os.path.join(path, fol, '*.jpg'))

    if len(all_i) < 3000:
        imgs.extend(all_i)
    
    else: 
        # shuffle the images
        random.shuffle(all_i)
        imgs.extend(all_i[:3000])


# shuffle all the images
import random
random.shuffle(imgs)
# imgs.sort()
model.classes = [0]
import os

# x0, y0, x1, y1, _, _ = results.xyxy[0][0].numpy().astype(int) 


from tqdm import tqdm
# for i in tqdm(range(0, len(imgs[:10000]), 32 )):
#     imgs_ = imgs[i:i+32]


# load the images in batches of 32
for i in tqdm(range(0, len(imgs), 128 * 4 )):
    imgs_ = imgs[i:i+(128 * 4)]

    results = model(imgs_, size=640)

    results_len = len(results.xyxy)

    os.makedirs('cropped_images', exist_ok=True)

    for no, img_name in enumerate(imgs_):

        try:
        # if True:
            img = cv2.imread(img_name)

            x0, y0, x1, y1, _, _ = results.xyxy[no][0].cpu().numpy().astype(int)

            width = x1 - x0
            height = y1 - y0

            if width > height:
                y0 = y0 - (width - height) // 2
                y1 = y1 + (width - height) // 2
            else:
                x0 = x0 - (height - width) // 2
                x1 = x1 + (height - width) // 2

            # pad 10 pixels to the bounding box
            x0 = x0 - 5
            x1 = x1 + 5
            y0 = y0 - 5
            y1 = y1 + 5

            # check if the coordinates are out of bounds
            if x0 < 0:
                x0 = 0
            if x1 > img.shape[1]:
                x1 = img.shape[1]
            if y0 < 0:
                y0 = 0
            if y1 > img.shape[0]:
                y1 = img.shape[0]

            # img = cv2.rectangle(img, (x0, y0), (x1, y1), (0, 255, 0), 2)

            # skeleton path
            skeleton_path = img_name.replace('output', 'input')
            skeleton_img = cv2.imread(skeleton_path)
            # skeleton_img = cv2.rectangle(skeleton_img, (x0, y0), (x1, y1), (0, 255, 0), 2)

            # crop the images
            img = img[y0:y1, x0:x1]
            skeleton_img = skeleton_img[y0:y1, x0:x1]

            img_name = img_name.replace('image_dataset', 'cropped_images')
            skeleton_path = skeleton_path.replace('image_dataset', 'cropped_images')

            os.makedirs(os.path.dirname(img_name), exist_ok=True)
            os.makedirs(os.path.dirname(skeleton_path), exist_ok=True)

            # save the images to the same path
            cv2.imwrite(img_name, img)  
            cv2.imwrite(skeleton_path, skeleton_img)
        
        except Exception as e:
            # print('Error with {}'.format(img_name))

            # print( e)

            pass

from logzero import logger
logger.info('Done with images')