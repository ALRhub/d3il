import os
import numpy as np
import random
import pickle

import gym
import gym_sorting

num_box = 6

env = gym.make('sorting-v0', render=False, num_boxes=num_box)

random.seed(0)
np.random.seed(0)

test_contexts = []
for i in range(1200):

    context = env.manager.sample()
    test_contexts.append(context)

with open(str(num_box) + "_test_contexts.pkl", "wb") as f:
    pickle.dump(test_contexts, f)


data_dir = str(num_box) + "_boxes/state/"
file_lists = np.load(str(num_box) + "_boxes_train_files.pkl", allow_pickle=True)

train_contexts = []

for file in file_lists[:60]:

    arr = np.load(data_dir + file, allow_pickle=True,)

    train_contexts.append(arr["context"])

with open(str(num_box) + "_train_contexts.pkl", "wb") as f:
    pickle.dump(train_contexts, f)