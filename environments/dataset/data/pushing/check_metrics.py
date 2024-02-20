import os
import numpy as np
import random
import pickle

file_lists = os.listdir("all_data")

random.shuffle(file_lists)


def rotation_distance(p: np.array, q: np.array):
    """
    Calculates the rotation angular between two quaternions
    param p: quaternion
    param q: quaternion
    theta: rotation angle between p and q (rad)
    """
    assert p.shape == q.shape, "p and q should be quaternion"
    theta = 2 * np.arccos(abs(p @ q))
    return theta

z_pos = []

pos_diff = []
quat_diff = []

robot_box_dists = []

lengths = []

# file_lists = np.load("train_files.pkl", allow_pickle=True)

for file in file_lists:

    arr = np.load("all_data/" + file, allow_pickle=True,)

    lengths.append(len(arr['robot']['c_pos']))

    red_box_pos = arr['red-box']['pos'][-1, :2]
    red_box_quat = arr['red-box']['quat']

    green_box_pos = arr['green-box']['pos'][-1, :2]
    green_box_quat = arr['green-box']['quat']

    red_target_pos = arr['red-target']['pos'][-1, :2]
    green_target_pos = arr['green-target']['pos'][-1, :2]

    pos_diff.append(min(np.linalg.norm(red_box_pos-red_target_pos), np.linalg.norm(red_box_pos-green_target_pos)))
    pos_diff.append(min(np.linalg.norm(green_box_pos-red_target_pos), np.linalg.norm(green_box_pos-green_target_pos)))


lengths = np.array(lengths)

print("data points: ", np.sum(lengths))

print('mean: ', np.mean(lengths))
print('std: ', np.std(lengths))
print('max: ', np.max(lengths))
print('min: ', np.min(lengths))

pos_diff = np.array(pos_diff)
a= 0