import fiftyone as fo
import fiftyone.zoo as foz
import os
from glob import glob
import fiftyone.core.utils as fou
import fiftyone.brain as fob
import fiftyone.brain.internal.models as fbm
import shutil
from tqdm import tqdm


folds = os.listdir("/home/cilab/teja/sign_langauge_dataset/HumanPoser/cropped_images_segmented/output")

# for fold in tqdm(folds):

def get_unq(fold):
    # '/home/cilab/teja/sign_langauge_dataset/HumanPoser/cropped_images_segmented/output/1176566_1b1'
    dataset_dir_imgs = glob(f"/home/cilab/teja/sign_langauge_dataset/HumanPoser/cropped_images_segmented/output/{fold}/*.jpg")


    if len(dataset_dir_imgs) < 300:
        return

    dataset = fo.Dataset.from_images(
        dataset_dir_imgs
    )

    model = foz.load_zoo_model("clip-vit-base32-torch")
    embeddings = dataset.compute_embeddings(model, batch_size=16)
    results = fob.compute_similarity(
        dataset, embeddings=embeddings, brain_key="img_sim"
    )
    results.find_unique(1500)
    unique_view = dataset.select(results.unique_ids)
    # print all the names of the images
    for idx, sample in enumerate(unique_view):
        # print(sample.filepath)
        og_path = sample.filepath
        og_path_update = og_path.replace('cropped_images_segmented', 'unq_dataset')
        os.makedirs(os.path.dirname(og_path_update), exist_ok=True)
        shutil.copy(og_path, og_path_update)

        sk_path = og_path.replace('output', 'input')
        sk_path_update = sk_path.replace('cropped_images_segmented', 'unq_dataset')
        os.makedirs(os.path.dirname(sk_path_update), exist_ok=True)
        shutil.copy(sk_path, sk_path_update)



import multiprocessing as mp
from multiprocessing import Pool

if __name__ == '__main__':

    # use mutliprocessing to speed up the process
    # pool = Pool(mp.cpu_count())
    # pool.map(get_unq, folds)
    for fold in tqdm(folds):
        get_unq(fold)