import os
import pickle
import random
import glob
from agents.utils.sim_path import sim_framework_path

random.seed(42)

data_dirs = ["4_boxes/state"]
             #"4_boxes/state"]
             # "6_boxes/state"]

for data_dir in data_dirs:

    basename = data_dir.split('/')[0]

    train_files = []
    eval_files = []

    all_data = os.listdir(data_dir)

    random.shuffle(all_data)

    num_split = int(len(all_data) * 0.1)

    train_files += all_data[num_split:]
    eval_files += all_data[:num_split]

    with open(basename + "_train_files.pkl", "wb") as f:
        pickle.dump(train_files, f)

    with open(basename + "_eval_files.pkl", "wb") as f:
        pickle.dump(eval_files, f)