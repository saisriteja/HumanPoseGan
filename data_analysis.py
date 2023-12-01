# after creating the cropped images we are do a dataset viz

from glob import glob
from tqdm import tqdm
import os

# path = './image_dataset/output'

path = './unq_dataset/output'

img_info = dict()

for fol in os.listdir(path):
    img_info[fol] = len(glob(os.path.join(path, fol, '*.jpg')))


# sort the dict by values
img_info = {k: v for k, v in sorted(img_info.items(), key=lambda item: item[1])}

# pritn the sorted dict
for k, v in img_info.items():
    print('{}: {}'.format(k, v))

# print length of the dict
print(len(img_info))


