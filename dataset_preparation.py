from src.person_manipulation import generate_segmentation_vid, get_pose
from src.video_utils import crop_video
from src.get_keypoints import generate_skeleton_pose
from logzero import logger
import toml
import shutil
import os
import glob
from multiprocessing import Pool
from tqdm import tqdm
import time

class BuildDatabase:
    def __init__(self, config) -> None:
        self.config = config

        # check if the vidpath is a directory or a file
        if os.path.isdir(self.config["vidpath"]):
            all_files = glob.glob(os.path.join(self.config["vidpath"], "*.mp4"))
            num_cores = os.cpu_count()

            start_time = time.time()

            # use multi threading to run the code
            with Pool(num_cores) as p:
                p.map(self.run, all_files)

            end_time = time.time()

            logger.info(f"Time taken to run the code is {end_time - start_time}")
        else:
            self.run(self.config["vidpath"])    

    def run(self, vidpath):
        xmin = self.config["xmin"]
        ymin = self.config["ymin"]
        extension = self.config["extension"]
        ratio = self.config["ratio"]
        self.name = self.config["name"]
        self.build_segmentation(vidpath)
        self.build_cropped_person(vidpath, xmin, ymin, extension)
        self.build_skeleton_pose(vidpath)
        self.build_skeleton_vid(vidpath)

        os.remove(vidpath.split(".")[0] + "_segmented.mp4")

    def build_segmentation(self, vidpath):
        try:
            # vidpath = 'dataset/output.mp4'
            # generate_segmentation_vid(vidpath)
            seg_vidpath = vidpath.split(".")[0] + "_segmented.mp4"
            shutil.copy(vidpath, seg_vidpath)
        except Exception as e:
            logger.error(f"The error is {e}")

    def build_cropped_person(self, vidpath, xmin, ymin, extension):
        try:
            vidpath = vidpath.split(".")[0] + "_segmented.mp4"
            outputpath = vidpath.split(".")[0] + "_cropped.mp4"

            if extension != 0:
                crop_video(vidpath, outputpath, xmin, ymin, extension)

            else:
                shutil.copy(vidpath, outputpath)
                logger.debug("No cropping done")
        except Exception as e:
            logger.error(f"The error is {e}")

    def build_skeleton_pose(self, vidpath):
        try:
            vidpath = vidpath.split(".")[0] + "_segmented_cropped.mp4"
            posepath = vidpath.split(".")[0] + ".pose"
            get_pose(vidpath, posepath)

        except Exception as e:
            logger.error(f"The error is {e}")

    def build_skeleton_vid(self, vidpath):
        try:
            skeleton_output = vidpath.split(".")[0] + "_skeleton.mp4"
            posepath2 = vidpath.split(".")[0] + "_segmented_cropped.pose"
            posepath1 = "utils/essen.pose"
            generate_skeleton_pose(posepath1, posepath2, skeleton_output)

            os.remove(vidpath.split(".")[0] + "_segmented_cropped.mp4")
        except Exception as e:
            logger.error(f"The error is {e}")



if __name__ == "__main__":
    config = toml.load("config_dataset.toml")
    build_database = BuildDatabase(config)
    build_database.run()
