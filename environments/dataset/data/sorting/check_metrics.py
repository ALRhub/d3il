import os
import numpy as np
import random
import pickle

data_dir = "2_boxes/state/"
file_lists = os.listdir(data_dir)

random.shuffle(file_lists)

red_target_pos = np.array([0.4, 0.32])
blue_target_pos = np.array([0.625, 0.32])

check_finished_red = []
check_finished_blue = []

lengths = []

# file_lists = np.load("6_boxes_train_files.pkl", allow_pickle=True)

for file in file_lists:

    arr = np.load(data_dir + file, allow_pickle=True,)

    lengths.append(len(arr['red-box1']['pos']))

    # red_1 = arr['red-box1']['pos'][-1][:2]
    # red_2 = arr['red-box2']['pos'][-1][:2]
    # red_3 = arr['red-box3']['pos'][-1][:2]
    #
    # blue_1 = arr['blue-box1']['pos'][-1][:2]
    # blue_2 = arr['blue-box2']['pos'][-1][:2]
    # blue_3 = arr['blue-box3']['pos'][-1][:2]
    #
    # red = np.vstack((red_1, red_2, red_3))
    #
    # blue = np.vstack((blue_1, blue_2, blue_3))
    #
    # if (red[:, 0] > 0.3).all() and (red[:, 0] < 0.5).all() and (red[:, 1] > 0.22).all() and (red[:, 1] < 0.41).all():
    #     check_finished_red.append(True)
    # else:
    #     check_finished_red.append(False)
    #     print(file)
    #
    # if (blue[:, 0] > 0.525).all() and (blue[:, 0] < 0.725).all() and (blue[:, 1] > 0.22).all() and (blue[:, 1] < 0.41).all():
    #     check_finished_blue.append(True)
    # else:
    #     check_finished_blue.append(False)
    #     print(file)

lengths = np.array(lengths)

print("data points: ", np.sum(lengths))

print('mean: ', np.mean(lengths))
print('std: ', np.std(lengths))
print('max: ', np.max(lengths))
print('min: ', np.min(lengths))


check_finished_red = np.array(check_finished_red)
check_finished_blue = np.array(check_finished_blue)

a=0