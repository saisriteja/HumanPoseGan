from glob import glob
import os

allmp4 = glob("/home/cilab/teja/sign_langauge_dataset/output_vids/*.mp4")


rm_files = []
for f in allmp4:
    if 'segment' in f:
        rm_files.append(f)
from tqdm import tqdm
for f in tqdm(rm_files):
    os.remove(f)