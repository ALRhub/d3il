import os
import numpy as np
import random
import pickle

from tqdm import tqdm


def gen_modes(data_dir, train_files, num_box=2):

    file_lists = np.load(train_files, allow_pickle=True)

    # random.shuffle(file_lists)

    red_target_pos = np.array([0.4, 0.32])
    blue_target_pos = np.array([0.625, 0.32])

    all_modes = {}

    for file in tqdm(file_lists):

        arr = np.load(data_dir + file, allow_pickle=True,)

        mode = np.array([0, 0, 0, 0, 0, 0])
        mode_step = 0
        min_inds = []

        for i in range(len(arr['red-box1']['pos'])):

            if mode_step > 5:
                break

            if num_box == 2:

                red_box_1_pos = arr['red-box1']['pos'][i][:2]
                blue_box_1_pos = arr['blue-box1']['pos'][i][:2]

                box_pos = np.vstack((red_box_1_pos, blue_box_1_pos))

                red_dist = np.linalg.norm(box_pos[:1] - np.reshape(red_target_pos, (1, -1)), axis=-1)
                blue_dist = np.linalg.norm(box_pos[1:] - np.reshape(blue_target_pos, (1, -1)), axis=-1)

                dists = np.concatenate((red_dist, blue_dist))
                dists[min_inds] = 100000

                min_ind = np.argmin(dists)
                min_box_pos = box_pos[min_ind]

                if min_ind < 1:
                    # manipulate red box, 0
                    if_finish = min_box_pos[0] > 0.3 and min_box_pos[0] < 0.5 and min_box_pos[1] > 0.22 and min_box_pos[
                        1] < 0.41

                    if if_finish:
                        mode[mode_step] = 0
                        mode_step += 1

                        min_inds.append(min_ind)
                else:
                    # manipulate blue box
                    if_finish = min_box_pos[0] > 0.525 and min_box_pos[0] < 0.725 and min_box_pos[1] > 0.22 and \
                                min_box_pos[1] < 0.41

                    if if_finish:
                        mode[mode_step] = 1
                        mode_step += 1

                        min_inds.append(min_ind)

            elif num_box == 4:
                red_box_1_pos = arr['red-box1']['pos'][i][:2]
                red_box_2_pos = arr['red-box2']['pos'][i][:2]

                blue_box_1_pos = arr['blue-box1']['pos'][i][:2]
                blue_box_2_pos = arr['blue-box2']['pos'][i][:2]

                box_pos = np.vstack((red_box_1_pos, red_box_2_pos, blue_box_1_pos, blue_box_2_pos))

                red_dist = np.linalg.norm(box_pos[:2] - np.reshape(red_target_pos, (1, -1)), axis=-1)
                blue_dist = np.linalg.norm(box_pos[2:] - np.reshape(blue_target_pos, (1, -1)), axis=-1)

                dists = np.concatenate((red_dist, blue_dist))
                dists[min_inds] = 100000

                min_ind = np.argmin(dists)
                min_box_pos = box_pos[min_ind]

                if min_ind < 2:
                    # manipulate red box, 0
                    if_finish = min_box_pos[0] > 0.3 and min_box_pos[0] < 0.5 and min_box_pos[1] > 0.22 and min_box_pos[1] < 0.41

                    if if_finish:
                        mode[mode_step] = 0
                        mode_step += 1

                        min_inds.append(min_ind)
                else:
                    # manipulate blue box
                    if_finish = min_box_pos[0] > 0.525 and min_box_pos[0] < 0.725 and min_box_pos[1] > 0.22 and min_box_pos[1] < 0.41

                    if if_finish:
                        mode[mode_step] = 1
                        mode_step += 1

                        min_inds.append(min_ind)

            elif num_box == 6:

                red_box_1_pos = arr['red-box1']['pos'][i][:2]
                red_box_2_pos = arr['red-box2']['pos'][i][:2]
                red_box_3_pos = arr['red-box3']['pos'][i][:2]

                blue_box_1_pos = arr['blue-box1']['pos'][i][:2]
                blue_box_2_pos = arr['blue-box2']['pos'][i][:2]
                blue_box_3_pos = arr['blue-box3']['pos'][i][:2]

                box_pos = np.vstack((red_box_1_pos, red_box_2_pos, red_box_3_pos, blue_box_1_pos, blue_box_2_pos, blue_box_3_pos))

                red_dist = np.linalg.norm(box_pos[:3] - np.reshape(red_target_pos, (1, -1)), axis=-1)
                blue_dist = np.linalg.norm(box_pos[3:] - np.reshape(blue_target_pos, (1, -1)), axis=-1)

                dists = np.concatenate((red_dist, blue_dist))
                dists[min_inds] = 100000

                min_ind = np.argmin(dists)
                min_box_pos = box_pos[min_ind]

                if min_ind < 3:
                    # manipulate red box, 0
                    if_finish = min_box_pos[0] > 0.3 and min_box_pos[0] < 0.5 and min_box_pos[1] > 0.22 and min_box_pos[1] < 0.41

                    if if_finish:
                        mode[mode_step] = 0
                        mode_step += 1

                        min_inds.append(min_ind)
                else:
                    # manipulate blue box
                    if_finish = min_box_pos[0] > 0.525 and min_box_pos[0] < 0.725 and min_box_pos[1] > 0.22 and min_box_pos[1] < 0.41

                    if if_finish:
                        mode[mode_step] = 1
                        mode_step += 1

                        min_inds.append(min_ind)

        if len(min_inds) != 6:
            print(file)

        mode = mode[:num_box]
        mode = int(np.packbits(mode)[0])

        keys = all_modes.keys()

        if mode not in keys:
            all_modes[mode] = 1
        else:
            all_modes[mode] += 1

        # print(mode, min_inds)

    with open(str(num_box) + "_modes.pkl", "wb") as f:
        pickle.dump(all_modes, f)

# delete env_0369_00.pkl, env_0268_00.pkl, env_0216_00.pkl

# gen_modes(data_dir="6_boxes/state/", train_files="6_boxes_train_files.pkl", num_box=6)

modes = np.load("4_modes.pkl", allow_pickle=True)

keys = modes.keys()
mode_prob = dict()

num_files = len(np.load("6_boxes_train_files.pkl", allow_pickle=True)) - 3

for key in keys:

    mode_prob[key] = modes[key] / num_files

with open("6_mode_prob.pkl", "wb") as f:
    pickle.dump(mode_prob, f)