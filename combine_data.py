import os
from glob import glob
from tqdm import tqdm
import multiprocessing as mp
from multiprocessing import Pool
def make_input_output_dirs(vidpath = "/home/cilab/teja/sign_langauge_dataset/viddata/*.mp4"):
    """
    Make input and output directories for the dataset"""

    allmp4 = glob(vidpath)[:10]
    os.makedirs('./viddata_combined', exist_ok=True)

    og = [ vid for vid in allmp4 if 'skeleton' not in vid]



    def frames_from_vid(vidpath, imgdirpath):
        os.makedirs(imgdirpath, exist_ok=True)
        os.system(f'ffmpeg -i {vidpath} -vf fps=30 {imgdirpath}/%07d.jpg')


    def get_frames_dataset(vidpath):
        try:
            skeleton_path = vidpath.replace('.mp4', '_skeleton.mp4')
            frames_from_vid(vidpath, './viddata_combined/' + vidpath.split('/')[-1].replace('.mp4', '') + '/output')
            frames_from_vid(skeleton_path, './viddata_combined/' + vidpath.split('/')[-1].replace('.mp4', '') + '/input')
        except Exception as e:
            print(e)



    # num_cores = mp.cpu_count()
    # with Pool(num_cores) as p:
    #     p.map(get_frames_dataset, og)

    for vid in tqdm(og):
        get_frames_dataset(vid)



make_input_output_dirs()

