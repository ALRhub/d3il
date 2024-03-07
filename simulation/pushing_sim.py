import logging
import multiprocessing as mp
import os
import random

import numpy as np
import torch
import wandb
from envs.gym_pushing_env.gym_pushing.envs.pushing import Block_Push_Env

from agents.utils.sim_path import sim_framework_path
from simulation.base_sim import BaseSim

log = logging.getLogger(__name__)


train_contexts = np.load(sim_framework_path("environments/dataset/data/pushing/train_contexts.pkl"),
                         allow_pickle=True)

test_contexts = np.load(sim_framework_path("environments/dataset/data/pushing/test_contexts.pkl"),
                        allow_pickle=True)


def assign_process_to_cpu(pid, cpus):
    os.sched_setaffinity(pid, cpus)


class Pushing_Sim(BaseSim):
    def __init__(
            self,
            seed: int,
            device: str,
            render: bool,
            n_cores: int = 1,
            n_contexts: int = 30,
            n_trajectories_per_context: int = 1,
    ):
        super().__init__(seed, device, render, n_cores)

        self.n_contexts = n_contexts
        self.n_trajectories_per_context = n_trajectories_per_context

    def eval_agent(self, agent, contexts, n_trajectories, mode_encoding, successes, mean_distance, pid, cpu_set):

        print(os.getpid(), cpu_set)
        assign_process_to_cpu(os.getpid(), cpu_set)

        env = Block_Push_Env(render=self.render)
        env.start()

        random.seed(pid)
        torch.manual_seed(pid)
        np.random.seed(pid)

        for context in contexts:
            for i in range(n_trajectories):

                agent.reset()

                print(f'Context {context} Rollout {i}')
                # training contexts
                # env.manager.set_index(context)
                obs = env.reset(random=False, context=test_contexts[context])
                # test contexts
                # test_context = env.manager.sample()
                # obs = env.reset(random=False, context=test_context)

                pred_action = env.robot_state()
                fixed_z = pred_action[2:]
                done = False

                while not done:

                    obs = np.concatenate((pred_action[:2], obs))

                    pred_action = agent.predict(obs)
                    pred_action = pred_action[0] + obs[:2]

                    pred_action = np.concatenate((pred_action, fixed_z, [0, 1, 0, 0]), axis=0)

                    obs, reward, done, info = env.step(pred_action)

                mode_encoding[context, i] = torch.tensor(info['mode'])
                successes[context, i] = torch.tensor(info['success'])
                mean_distance[context, i] = torch.tensor(info['mean_distance'])

    ################################
    # we use multi-process for the simulation
    # n_contexts: the number of different contexts of environment
    # n_trajectories_per_context: test each context for n times, this is mostly used for multi-modal data
    # n_cores: the number of cores used for simulation
    ###############################
    def test_agent(self, agent):

        log.info('Starting trained model evaluation')

        mode_encoding = torch.zeros([self.n_contexts, self.n_trajectories_per_context]).share_memory_()
        successes = torch.zeros((self.n_contexts, self.n_trajectories_per_context)).share_memory_()
        mean_distance = torch.zeros((self.n_contexts, self.n_trajectories_per_context)).share_memory_()

        contexts = np.arange(self.n_contexts)

        workload = self.n_contexts // self.n_cores

        num_cpu = mp.cpu_count()
        cpu_set = list(range(num_cpu))

        # start = self.seed * 20
        # end = start + 20
        #
        # cpu_set = cpu_set[start:end]
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
                        "successes": successes,
                        "mean_distance": mean_distance,
                        "pid": i,
                        "cpu_set": set(cpu_set[i:i + 1])
                    },
                )
                print("Start {}".format(i))
                p.start()
                p_list.append(p)
            [p.join() for p in p_list]

        else:
            self.eval_agent(agent, contexts, self.n_trajectories_per_context, mode_encoding, successes, mean_distance, 0, set([0]))

        n_modes = 4

        success_rate = torch.mean(successes).item()
        mode_probs = torch.zeros([self.n_contexts, n_modes])
        if n_modes == 1:
            for c in range(self.n_contexts):
                mode_probs[c, :] = torch.tensor(
                    [sum(mode_encoding[c, successes[c, :] == 1] == 0) / self.n_trajectories_per_context])

        elif n_modes == 2:
            for c in range(self.n_contexts):
                mode_probs[c, :] = torch.tensor(
                    [sum(mode_encoding[c, successes[c, :] == 1] == 0) / self.n_trajectories_per_context,
                     sum(mode_encoding[c, successes[c, :] == 1] == 1) / self.n_trajectories_per_context])

        elif n_modes == 4:
            for c in range(self.n_contexts):
                mode_probs[c, :] = torch.tensor(
                    [sum(mode_encoding[c, successes[c, :] == 1] == 0) / self.n_trajectories_per_context,
                     sum(mode_encoding[c, successes[c, :] == 1] == 1) / self.n_trajectories_per_context,
                     sum(mode_encoding[c, successes[c, :] == 1] == 2) / self.n_trajectories_per_context,
                     sum(mode_encoding[c, successes[c, :] == 1] == 3) / self.n_trajectories_per_context])

        mode_probs /= (mode_probs.sum(1).reshape(-1, 1) + 1e-12)
        print(f'p(m|c) {mode_probs}')

        entropy = - (mode_probs * torch.log(mode_probs + 1e-12) / torch.log(
            torch.tensor(n_modes))).sum(1).mean()

        wandb.log({'score': 0.5 * (success_rate + entropy)})
        wandb.log({'Metrics/successes': success_rate})
        wandb.log({'Metrics/entropy': entropy})
        wandb.log({'Metrics/distance': mean_distance.mean().item()})

        print(f'Mean Distance {mean_distance.mean().item()}')
        print(f'Successrate {success_rate}')
        print(f'entropy {entropy}')

        return successes, mode_encoding, mean_distance