import os
import pickle
import random
import glob
from agents.utils.sim_path import sim_framework_path

import numpy as np

tests = np.load("train_files_0.1_.pkl", allow_pickle=True)

random.seed(42)

train_files = []
eval_files = []

inside = os.listdir("inside")
outside = os.listdir("outside")

all_data = [inside, outside]

splits = [0.1]

random.shuffle(inside)
random.shuffle(outside)

for split in splits:

    train_files = []

    for data in all_data:

        len_train = 50 + int(len(data[50:]) * split)

        train_files += data[50:len_train]
        eval_files += data[:50]

    with open("train_files_" + str(split) + "_.pkl", "wb") as f:
        pickle.dump(train_files, f)

    # with open("eval_files.pkl", "wb") as f:
    #     pickle.dump(eval_files, f)