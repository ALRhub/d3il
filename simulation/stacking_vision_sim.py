import logging
import os
import copy

from multiprocessing import Process
import multiprocessing as mp
import random

import cv2
import numpy as np
import torch
import hydra
import wandb

from simulation.base_sim import BaseSim
from agents.utils.sim_path import sim_framework_path

import gym
import gym_stacking

log = logging.getLogger(__name__)


def assign_process_to_cpu(pid, cpus):
    os.sched_setaffinity(pid, cpus)


class Stacking_Sim(BaseSim):
    def __init__(
            self,
            seed: int,
            device: str,
            render: bool,
            n_cores: int = 1,
            n_contexts: int = 30,
            n_trajectories_per_context: int = 1,
            if_vision: bool = False,
            max_steps_per_episode: int = 500
    ):
        super().__init__(seed, device, render, n_cores, if_vision)

        self.n_contexts = n_contexts
        self.n_trajectories_per_context = n_trajectories_per_context

        self.max_steps_per_episode = max_steps_per_episode

        self.test_contexts = np.load(sim_framework_path("environments/dataset/data/stacking/test_contexts.pkl"),
                                     allow_pickle=True)

        # pre-define the mode
        self.mode_1 = {'r': 0, 'g': 1, 'b': 2}
        self.mode_2 = {'rg': 0, 'rb': 1, 'gr': 2, 'gb': 3, 'br': 4, 'bg': 5}
        self.mode_3 = {'rgb': 0, 'rbg': 1, 'grb': 2, 'gbr': 3, 'brg': 4, 'bgr': 5}

        self.modes = np.load(sim_framework_path("environments/dataset/data/stacking/vision_mode_prob.pkl"),
                             allow_pickle=True)

        self.mode_encoding_3 = np.zeros(6)
        self.mode_encoding_2 = np.zeros(6)

        for m_key in self.mode_3.keys():

            self.mode_encoding_3[self.mode_3[m_key]] = self.modes[m_key]
            self.mode_encoding_2[self.mode_3[m_key]] = self.modes[m_key]

        self.mode_encoding_1 = np.array([self.mode_encoding_3[i] + self.mode_encoding_3[i+1] for i in [0, 2, 4]])

        self.mode_encoding_1 = torch.tensor(self.mode_encoding_1)
        self.mode_encoding_2 = torch.tensor(self.mode_encoding_2)
        self.mode_encoding_3 = torch.tensor(self.mode_encoding_3)

        # self.mode_keys = np.array(list(self.modes.keys()))
        # self.n_mode = len(self.mode_keys)
        #
        # self.mode_dict = {}
        #
        # mode_encoding = []
        #
        # for i in range(self.n_mode):
        #     mode_encoding.append(self.modes[self.mode_keys[i]])
        #     self.mode_dict[self.mode_keys[i]] = i + 100
        #
        # self.mode_encoding = torch.tensor(mode_encoding)
        #
        # #################################################################
        # self.box_1_modes = {'r':0, 'g':0, 'b':0}
        # for mode_key in self.mode_keys:
        #     self.box_1_modes[mode_key[0]] += self.modes[mode_key]
        #
        # self.box_1_mode_keys = list(self.box_1_modes.keys())
        #
        # mode_encoding_box1 = []
        # for i in range(3):
        #     mode_encoding_box1.append(self.box_1_modes[self.box_1_mode_keys[i]])
        #     self.mode_dict[self.box_1_mode_keys[i]] = i + 200
        #
        # self.mode_encoding_box1 = torch.tensor(mode_encoding_box1)

    def eval_agent(self, agent, contexts, n_trajectories, mode_encoding, mode_encoding_1_box, mode_encoding_2_box,
                   successes, successes_1, successes_2, cpu_set, pid, if_vision=False):

        print(os.getpid(), cpu_set)
        assign_process_to_cpu(os.getpid(), cpu_set)

        env = gym.make('stacking-v0', max_steps_per_episode=self.max_steps_per_episode, render=self.render, if_vision=if_vision)
        env.start()

        random.seed(pid)
        torch.manual_seed(pid)
        np.random.seed(pid)

        for context in contexts:

            sim_step = 0
            for i in range(n_trajectories):

                agent.reset()

                print(f'Context {context} Rollout {i}')
                # training contexts
                # env.manager.set_index(context)
                # obs = env.reset()
                # test contexts
                # test_context = env.manager.sample()
                obs = env.reset(random=False, context=self.test_contexts[context])

                if if_vision:

                    j_state, bp_image, inhand_image = obs
                    bp_image = bp_image.transpose((2, 0, 1)) / 255.
                    inhand_image = inhand_image.transpose((2, 0, 1)) / 255.

                    des_j_state = j_state
                    done = False

                    while not done:
                        pred_action = agent.predict((bp_image, inhand_image, des_j_state), if_vision=if_vision)[0]
                        pred_action[:7] = pred_action[:7] + des_j_state[:7]

                        obs, reward, done, info = env.step(pred_action)

                        des_j_state = pred_action

                        robot_pos, bp_image, inhand_image = obs

                        # cv2.imshow('0', bp_image)
                        # cv2.waitKey(1)
                        #
                        # cv2.imshow('1', inhand_image)
                        # cv2.waitKey(1)

                        bp_image = bp_image.transpose((2, 0, 1)) / 255.
                        inhand_image = inhand_image.transpose((2, 0, 1)) / 255.

                else:
                    pred_action, _, _ = env.robot_state()
                    pred_action = pred_action.astype(np.float32)

                    done = False
                    while not done:
                        # obs = np.concatenate((pred_action[:-1], obs))
                        # obs = np.concatenate((pred_action[:-1], obs, np.array([sim_step])))

                        obs = np.concatenate((pred_action, obs))

                        # print(obs[11], obs[15], obs[-1])

                        pred_action = agent.predict(obs)[0]
                        pred_action[:7] = pred_action[:7] + obs[:7]

                        # pred_action = np.concatenate((pred_action, [0, 1, 0, 0, 1]))
                        # j_pos = pred_action[:7]
                        # j_vel = pred_action[7:14]
                        # gripper_width = pred_action[14]

                        # print(gripper_width)

                        obs, reward, done, info = env.step(pred_action)
                        sim_step += 1


                # if info['mode'] not in self.mode_keys:
                #     pass
                # else:
                #     mode_encoding[context, i] = torch.tensor(self.mode_dict[info['mode']])

                if len(info['mode']) > 2:
                    mode_encoding_1_box[context, i] = torch.tensor(self.mode_1[info['mode'][:1]])
                    mode_encoding_2_box[context, i] = torch.tensor(self.mode_2[info['mode'][:2]])

                    mode_encoding[context, i] = torch.tensor(self.mode_3[info['mode'][:3]])

                elif len(info['mode']) > 1:
                    mode_encoding_1_box[context, i] = torch.tensor(self.mode_1[info['mode'][:1]])
                    mode_encoding_2_box[context, i] = torch.tensor(self.mode_2[info['mode'][:2]])

                elif len(info['mode']) > 0:
                    mode_encoding_1_box[context, i] = torch.tensor(self.mode_1[info['mode'][:1]])

                else:
                    pass

                successes[context, i] = torch.tensor(info['success'])
                successes_1[context, i] = torch.tensor(info['success_1'])
                successes_2[context, i] = torch.tensor(info['success_2'])
                # mean_distance[context, i] = torch.tensor(info['mean_distance'])

    def cal_KL(self, mode_encoding, successes, prior_encoding, n_mode=6):

        mode_probs = torch.zeros([self.n_contexts, n_mode])

        for c in range(self.n_contexts):

            for num in range(n_mode):

                mode_probs[c, num] = torch.tensor([sum(mode_encoding[c, successes[c, :] == 1] == num)
                                                   / self.n_trajectories_per_context])

        mode_probs /= (mode_probs.sum(1).reshape(-1, 1) + 1e-12)
        print(f'p(m|c) {mode_probs}')

        mode_probs = mode_probs[torch.nonzero(mode_probs.sum(1), as_tuple=True)[0]]

        entropy = - (mode_probs * torch.log(mode_probs + 1e-12) / torch.log(
            torch.tensor(n_mode))).sum(1).mean()

        log_ = (mode_probs * torch.log(prior_encoding + 1e-12) / torch.log(
            torch.tensor(n_mode))).sum(1).mean()

        KL = - entropy - log_

        return entropy, KL

    ################################
    # we use multi-process for the simulation
    # n_contexts: the number of different contexts of environment
    # n_trajectories_per_context: test each context for n times, this is mostly used for multi-modal data
    # n_cores: the number of cores used for simulation
    ###############################
    def test_agent(self, agent):

        log.info('Starting trained model evaluation')

        mode_encoding_1_box = torch.zeros([self.n_contexts, self.n_trajectories_per_context]).share_memory_()
        mode_encoding_2_box = torch.zeros([self.n_contexts, self.n_trajectories_per_context]).share_memory_()
        # mode encoding for 2 and 3 boxes are the same
        mode_encoding = torch.zeros([self.n_contexts, self.n_trajectories_per_context]).share_memory_()

        successes = torch.zeros((self.n_contexts, self.n_trajectories_per_context)).share_memory_()
        successes_1 = torch.zeros((self.n_contexts, self.n_trajectories_per_context)).share_memory_()
        successes_2 = torch.zeros((self.n_contexts, self.n_trajectories_per_context)).share_memory_()

        contexts = np.arange(self.n_contexts)

        workload = self.n_contexts // self.n_cores

        num_cpu = mp.cpu_count()
        cpu_set = list(range(num_cpu))

        print("there are cpus: ", num_cpu)

        ctx = mp.get_context('spawn')

        p_list = []
        if self.n_cores > 1:
            for i in range(self.n_cores):
                p = ctx.Process(
                    target=self.eval_agent,
                    kwargs={
                        "agent": agent,
                        "contexts": contexts[i * workload:(i + 1) * workload],
                        "n_trajectories": self.n_trajectories_per_context,
                        "mode_encoding": mode_encoding,
                        "mode_encoding_1_box": mode_encoding_1_box,
                        "mode_encoding_2_box": mode_encoding_2_box,
                        "successes": successes,
                        "successes_1": successes_1,
                        "successes_2": successes_2,
                        "pid": i,
                        "cpu_set": set(cpu_set[i:i + 1]),
                        "if_vision": self.if_vision
                    },
                )
                print("Start {}".format(i))
                p.start()
                p_list.append(p)
            [p.join() for p in p_list]

        else:
            self.eval_agent(agent, contexts, self.n_trajectories_per_context, mode_encoding, mode_encoding_1_box, mode_encoding_2_box,
                            successes, successes_1, successes_2, set([0]), 0, if_vision=self.if_vision)

        box1_success_rate = torch.mean(successes_1).item()
        box2_success_rate = torch.mean(successes_2).item()

        success_rate = torch.mean(successes).item()

        entropy_1, KL_1 = self.cal_KL(mode_encoding_1_box, successes_1, self.mode_encoding_1, n_mode=3)
        entropy_2, KL_2 = self.cal_KL(mode_encoding_2_box, successes_2, self.mode_encoding_2, n_mode=6)
        entropy_3, KL_3 = self.cal_KL(mode_encoding, successes, self.mode_encoding_3, n_mode=6)

        wandb.log({'score': (box1_success_rate + box2_success_rate + success_rate)})

        wandb.log({'Metrics/successes': success_rate})
        wandb.log({'Metrics/entropy_3': entropy_3})
        wandb.log({'Metrics/KL_3': KL_3})

        wandb.log({'Metrics/successes_1_box': box1_success_rate})
        wandb.log({'Metrics/entropy_1': entropy_1})
        wandb.log({'Metrics/KL_1': KL_1})

        wandb.log({'Metrics/successes_2_boxes': box2_success_rate})
        wandb.log({'Metrics/entropy_2': entropy_2})
        wandb.log({'Metrics/KL_2': KL_2})

        # print(f'Mean Distance {mean_distance.mean().item()}')
        print(f'Successrate {success_rate}')
        print(f'Successrate_1 {box1_success_rate}')
        print(f'Successrate_2 {box2_success_rate}')
        # print(f'entropy {entropy_1}')
        # print(f'KL {KL_1}')

        score = box1_success_rate + box2_success_rate + success_rate

        return score, mode_encoding#, mean_distance