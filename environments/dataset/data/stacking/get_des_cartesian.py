import os
import numpy as np
import gym
import gym_stacking
import pickle

# env = gym.make('stacking-v0')


file_lists = os.listdir("all_data")

# file_lists = np.load("eval_files.pkl", allow_pickle=True)

lengths = []
for file in file_lists:

    arr = np.load("all_data/" + file, allow_pickle=True)

    lengths.append(len(arr['robot']['c_pos']))
    # sv_arr = arr.copy()
    #
    # robot_des_j_pos = arr['robot']['des_j_pos']
    #
    # for i in range(len(robot_des_j_pos)):
    #
    #     sv_arr['robot']['des_c_pos'][i] = env.robot.getForwardKinematics(robot_des_j_pos[i])[0]
    #     sv_arr['robot']['des_c_quat'][i] = env.robot.getForwardKinematics(robot_des_j_pos[i])[1]
    #
    # sv_file = 'all_data_new/' + file
    #
    # with open(sv_file, "wb") as f:
    #     pickle.dump(sv_arr, f)

lengths = np.array(lengths)

print("data points: ", np.sum(lengths))

print('mean: ', np.mean(lengths))
print('std: ', np.std(lengths))
print('max: ', np.max(lengths))
print('min: ', np.min(lengths))