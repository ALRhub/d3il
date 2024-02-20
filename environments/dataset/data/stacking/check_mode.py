import os
import numpy as np
import random
import pickle
import matplotlib.pyplot as plt
from tqdm import tqdm


def gen_modes():
    modes = ['r', 'g', 'b']

    file_lists = np.load("vision_train_files.pkl", allow_pickle=True)

    # max length 965
    # lengths = []
    #
    # for file in file_lists:
    #     arr = np.load("all_data/" + file, allow_pickle=True)
    #
    #     lengths.append(len(arr['red-box']['pos']))

    all_modes = {}

    for file in tqdm(file_lists):

        arr = np.load("vision_data/state/" + file, allow_pickle=True)

        min_inds = []
        mode_encoding = []
        mode = ''

        for i in range(len(arr['robot']['c_pos'])):

            red_pos = arr['red-box']['pos'][i, :2]
            green_pos = arr['green-box']['pos'][i, :2]
            blue_pos = arr['blue-box']['pos'][i, :2]

            target_pos = arr['target-box']['pos'][0, :2]

            box_pos = np.vstack((red_pos, green_pos, blue_pos))

            dists = np.linalg.norm(box_pos - np.reshape(target_pos, (1, -1)), axis=-1)
            dists[min_inds] = 100000

            min_ind = np.argmin(dists)

            if dists[min_ind] <= 0.06:
                mode_encoding.append(modes[min_ind])
                min_inds.append(min_ind)

        mode = mode.join(mode_encoding)

        # mode = np.array(min_inds)
        # mode = int(np.packbits(mode)[0])

        if mode not in all_modes.keys():
            all_modes[mode] = 1
        else:
            all_modes[mode] += 1

        print(mode, min_inds)

    with open("vision_modes.pkl", "wb") as f:
        pickle.dump(all_modes, f)

    return all_modes


# all_modes = gen_modes()

modes = np.load("vision_modes.pkl", allow_pickle=True)

keys = modes.keys()
mode_prob = dict()

num_files = len(np.load("eval_files.pkl", allow_pickle=True))

for key in keys:

    mode_prob[key] = modes[key] / num_files

with open("vision_mode_prob.pkl", "wb") as f:
    pickle.dump(mode_prob, f)

#
# with open("all_modes.pkl", "wb") as f:
#     pickle.dump(all_modes, f)

# all_modes = np.load("all_modes.pkl", allow_pickle=True)
#
# for mode in all_modes.keys():
#
#     file_list = all_modes[mode]
#
#     fig, axs = plt.subplots(7)
#
#     for file in file_list:
#         arr = np.load("all_data/" + file, allow_pickle=True)
#
#         # (n, 7)
#         robot_des_j_pos = arr['robot']['des_j_pos']
#
#         if len(robot_des_j_pos) > 1000:
#             print(file)
#
#         for i in range(7):
#             axs[i].plot(robot_des_j_pos[:, i])
#
#
#     plt.show()
#
#     print(mode)