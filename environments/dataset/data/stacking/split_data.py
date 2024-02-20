import os
import numpy as np
import random
import pickle
import matplotlib.pyplot as plt

random.seed(42)

train_files = []
eval_files = []

all_data = os.listdir("vision_data/state")

random.shuffle(all_data)

# num_split = int(len(all_data) * 0.1)
num_split = 94

train_files += all_data[num_split:]
eval_files += all_data[:num_split]

with open("vision_train_files.pkl", "wb") as f:
    pickle.dump(train_files, f)

with open("vision_eval_files.pkl", "wb") as f:
    pickle.dump(eval_files, f)

# modes = ['r', 'g', 'b']
#
# file_lists = os.listdir("all_data")
#
# all_modes = {}
#
# for file in file_lists:
#
#     arr = np.load("joint_data/" + file, allow_pickle=True)
#
#     min_inds = []
#     mode_encoding = []
#     mode = ''
#
#     for i in range(len(arr['robot']['c_pos'])):
#
#         red_pos = arr['red-box']['pos'][i, :2]
#         green_pos = arr['green-box']['pos'][i, :2]
#         blue_pos = arr['blue-box']['pos'][i, :2]
#
#         target_pos = arr['target-box']['pos'][0, :2]
#
#         box_pos = np.vstack((red_pos, green_pos, blue_pos))
#
#         dists = np.linalg.norm(box_pos - np.reshape(target_pos, (1, -1)), axis=-1)
#         dists[min_inds] = 100000
#
#         min_ind = np.argmin(dists)
#
#         if dists[min_ind] <= 0.06:
#             mode_encoding.append(modes[min_ind])
#             min_inds.append(min_ind)
#
#         if len(min_inds) == 1:
#             break
#
#     if min_inds[0] == 2:
#
#         sv_arr = arr.copy()
#
#         keys = list(sv_arr.keys())
#         del keys[0]
#
#         for key in keys:
#
#             values = sv_arr[key]
#
#             for value_key in list(values.keys()):
#
#                 sv_arr[key][value_key] = sv_arr[key][value_key][:i+100]
#
#         sv_file = "joint_blue/" + file
#         with open(sv_file, "wb") as f:
#             pickle.dump(sv_arr, f)