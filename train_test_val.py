import splitfolders
import shutil
import os
os.makedirs("gan_train_test_val",exist_ok=True)

# move the folder into the dire
shutil.move("./gan_dataset", "./gan_train_test_val/gan_dataset")

splitfolders.ratio("./gan_train_test_val", output="gan_training_dataset",seed=1337, ratio=(.8, .1, .1), group_prefix=None, move=False)