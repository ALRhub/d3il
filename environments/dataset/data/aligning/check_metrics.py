import os
import numpy as np
import random
import pickle

data_dir = 'all_data/state/'
file_lists = os.listdir(data_dir)

# file_lists = np.load("eval_files.pkl", allow_pickle=True)

lengths = []
for file in file_lists:
    arr = np.load(data_dir + file, allow_pickle=True)

    lengths.append(len(arr['robot']['c_pos']))

lengths = np.array(lengths)

print("data points: ", np.sum(lengths))
print('mean: ', np.mean(lengths))
print('std: ', np.std(lengths))
print('max: ', np.max(lengths))
print('min: ', np.min(lengths))


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

for file in file_lists:

    arr = np.load("inside/" + file, allow_pickle=True,)

    robot_des_pos = arr['robot']['des_c_pos']

    z_pos.append(np.min(robot_des_pos[:, -1]))

    robot_c_pos = arr['robot']['c_pos'][-1]

    push_box_pos = arr['push-box']['pos'][-1]
    push_box_quat = arr['push-box']['quat'][-1]

    target_box_pos = arr['target-box']['pos'][-1]
    target_box_quat = arr['target-box']['quat'][-1]

    robot_box_dist = np.linalg.norm(push_box_pos[:2] - robot_c_pos[:2])

    robot_box_dists.append(robot_box_dist)

    box_goal_pos_dist = np.linalg.norm(push_box_pos - target_box_pos)
    box_goal_rot_dist = rotation_distance(push_box_quat, target_box_quat) / np.pi

    if robot_box_dist > 0.05:
        print(file)

    pos_diff.append(box_goal_pos_dist)
    quat_diff.append(box_goal_rot_dist)

z_pos = np.array(z_pos)
pos_diff = np.array(pos_diff)
quat_diff = np.array(quat_diff)

robot_box_dists = np.array(robot_box_dists)

a= 0
