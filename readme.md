# This is a human poser repo

## Making a virtual env

The code is tested with python 3.9.

```bash
python3 -m venv venv
source venv/bin/activate
bash install.txt
```

## Dataset Preparation

update the toml file.

```bash
vidpath = 'julia/signal-2023-06-28-00-35-42-312.mp4'   -> vidpath
name = 'julia' -> unq name of the person
xmin = 160 -> x left top
ymin = 0   -> y left top
extension = 360  -> extension of the box( x_min, y_min, x_min+extension, y_min+extension)
ratio = [0.5, 0.3, 0.2] -> ratio of train, val and test
```

run the command to generate a database of skeleton and images.

```
python
python dataset_preparation.py
```


```
python
python image_dir_making.py
python person_detection.py
python human_segmentation.py
python duplicate_removal.py
python gan_dataset_prep.py
python train_test_val.py 
```