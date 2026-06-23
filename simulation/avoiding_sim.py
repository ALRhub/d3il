import logging
import multiprocessing as mp
import os
import random
from envs.gym_avoiding_env.gym_avoiding.envs.avoiding import ObstacleAvoidanceEnv

import numpy as np
import torch
import wandb

from simulation.base_sim import BaseSim  # pins MUJOCO_GL=egl on import

log = logging.getLogger(__name__)


def assign_process_to_cpu(pid, cpus):
    os.sched_setaffinity(pid, cpus)


class Avoiding_Sim(BaseSim):
    def __init__(
            self,
            seed: int,
            device: str,
            render: bool,
            n_cores: int = 1,
            n_trajectories: int = 30,
            # --- visualisation hooks (all optional; default = current behaviour) ---
            agent_name: str = "agent",        # used to name saved artifacts per method
            save_traj_dir: str = None,        # if set, dump end-effector trajectories (.npz)
            record_video_dir: str = None,     # if set, render rollout videos (.mp4)
            record_n_videos: int = 0,         # number of rollouts to record
            record_backend: str = "egl",   # deprecated/ignored: rendering is always EGL
            record_size: int = 480,           # rendered frame size (square)
    ):
        super().__init__(seed, device, render, n_cores)

        self.n_trajectories = n_trajectories
        self.agent_name = agent_name
        # Hydra chdir's into a per-run dir, so resolve relative output dirs
        # against the original launch cwd -> all methods share one location.
        self.save_traj_dir = self._abs_dir(save_traj_dir)
        self.record_video_dir = self._abs_dir(record_video_dir)
        self.record_n_videos = record_n_videos
        self.record_backend = record_backend
        self.record_size = record_size

    @staticmethod
    def _abs_dir(path):
        if path is None or os.path.isabs(path):
            return path
        try:
            from hydra.utils import get_original_cwd
            return os.path.join(get_original_cwd(), path)
        except Exception:
            return os.path.abspath(path)

    def eval_agent(self, agent, n_trajectories, mode_encoding, successes, robot_c_pos, traj_lens, pid, cpu_set):

        print(os.getpid(), cpu_set)
        assign_process_to_cpu(os.getpid(), cpu_set)

        env = ObstacleAvoidanceEnv(render=self.render)
        env.start()

        random.seed(pid)
        torch.manual_seed(pid)
        np.random.seed(pid)

        for i in range(n_trajectories):

            agent.reset()

            print(f'core {pid}, Rollout {i}')

            obs = env.reset()

            pred_action = env.robot_state()
            fixed_z = pred_action[2:]
            done = False

            c_pos = [env.robot.current_c_pos]

            while not done:

                obs = np.concatenate((pred_action[:2], obs))

                pred_action = agent.predict(obs)
                pred_action = pred_action[0] + obs[:2]

                pred_action = np.concatenate((pred_action, fixed_z, [0, 1, 0, 0]), axis=0)

                obs, reward, done, info = env.step(pred_action)

                c_pos.append(env.robot.current_c_pos)

            c_pos = torch.tensor(np.array(c_pos))[:, :2]
            # Episodes can run longer than the fixed buffer width (rollouts go
            # up to max_steps_per_episode); clip so the assignment never errors.
            length = min(c_pos.shape[0], robot_c_pos.shape[1])
            robot_c_pos[pid * n_trajectories + i, :length, :] = c_pos[:length]
            traj_lens[pid * n_trajectories + i] = length

            mode_encoding[pid * n_trajectories + i, :] = torch.tensor(info[0])
            successes[pid * n_trajectories + i] = torch.tensor(info[1])

    ################################
    # we use multi-process for the simulation
    # n_trajectories: rollout policy for n times
    # n_cores: the number of cores used for simulation
    ###############################
    def test_agent(self, agent):

        log.info('Starting trained model evaluation')

        # 250 matches the env's max_steps_per_episode (+1 for the initial pos);
        # the assignment in eval_agent also clips defensively against overflow.
        robot_c_pos = torch.zeros([self.n_trajectories, 251, 2]).share_memory_()

        mode_encoding = torch.zeros([self.n_trajectories, 9]).share_memory_()
        successes = torch.zeros(self.n_trajectories).share_memory_()
        # actual (unpadded) length of each saved trajectory, for plotting
        traj_lens = torch.zeros(self.n_trajectories).share_memory_()

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
                        "n_trajectories": self.n_trajectories // self.n_cores,
                        "mode_encoding": mode_encoding,
                        "successes": successes,
                        "robot_c_pos": robot_c_pos,
                        "traj_lens": traj_lens,
                        "pid": i,
                        "cpu_set": set(cpu_set[i:i + 1])
                    },
                )
                print("Start {}".format(i))
                p.start()
                p_list.append(p)
            [p.join() for p in p_list]

        else:
            self.eval_agent(agent, self.n_trajectories, mode_encoding, successes, robot_c_pos, traj_lens, 0, set([0]))

        # Optionally dump the end-effector trajectories for cross-method plots.
        if self.save_traj_dir is not None:
            self._save_trajectories(robot_c_pos, mode_encoding, successes, traj_lens)

        # Optionally render rollout videos for visual comparison.
        if self.record_video_dir is not None and self.record_n_videos > 0:
            self._record_videos(agent)

        success_rate = torch.mean(successes).item()

        # calculate entropy
        data = mode_encoding[successes == 1].numpy()
        data_decimal = data.dot(1 << np.arange(data.shape[-1]))
        _, counts = np.unique(data_decimal, return_counts=True)
        mode_dist = counts / np.sum(counts)
        entropy = - np.sum(mode_dist * (np.log(mode_dist) / np.log(24)))

        wandb.log({'score': (success_rate * 0.8 + entropy * 0.2)})
        wandb.log({'Metrics/successes': success_rate})
        wandb.log({'Metrics/entropy': entropy})

        print(f'Successrate {success_rate}')
        print(f'entropy {entropy}')

        return successes, entropy

    # ------------------------------------------------------------------
    # Visualisation helpers
    # ------------------------------------------------------------------
    def _save_trajectories(self, robot_c_pos, mode_encoding, successes, traj_lens):
        """Save per-rollout end-effector trajectories as <agent_name>_seed<seed>.npz.

        ``scripts/plot_avoiding_trajectories.py`` consumes these files to draw a
        side-by-side comparison of the methods over the obstacle field.
        """
        os.makedirs(self.save_traj_dir, exist_ok=True)
        path = os.path.join(self.save_traj_dir, f"{self.agent_name}_seed{self.seed}.npz")
        np.savez_compressed(
            path,
            c_pos=robot_c_pos.cpu().numpy(),        # (N, 251, 2), zero-padded
            lengths=traj_lens.cpu().numpy().astype(np.int32),  # (N,) real lengths
            modes=mode_encoding.cpu().numpy(),      # (N, 9) one-hot-ish mode id
            successes=successes.cpu().numpy(),      # (N,)
            agent_name=self.agent_name,
            seed=self.seed,
        )
        log.info(f"Saved {robot_c_pos.shape[0]} trajectories to {path}")

    def _record_videos(self, agent):
        """Render a few rollouts offscreen (headless) into one .mp4 per method."""
        # Rendering is pinned to EGL (see module top); enforce it again here so
        # it holds even if the process environment was changed elsewhere.
        os.environ["MUJOCO_GL"] = "egl"
        import imageio

        os.makedirs(self.record_video_dir, exist_ok=True)
        path = os.path.join(self.record_video_dir, f"{self.agent_name}_seed{self.seed}.mp4")

        env = ObstacleAvoidanceEnv(render=False)
        env.start()
        random.seed(self.seed)
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        sz = self.record_size
        writer = imageio.get_writer(path, fps=30)
        try:
            for i in range(self.record_n_videos):
                agent.reset()
                obs = env.reset()
                pred_action = env.robot_state()
                fixed_z = pred_action[2:]
                done = False
                while not done:
                    obs = np.concatenate((pred_action[:2], obs))
                    pred_action = agent.predict(obs)
                    pred_action = pred_action[0] + obs[:2]
                    pred_action = np.concatenate((pred_action, fixed_z, [0, 1, 0, 0]), axis=0)
                    obs, reward, done, info = env.step(pred_action)
                    frame = env.bp_cam.get_image(width=sz, height=sz, depth=False)
                    writer.append_data(np.asarray(frame, dtype=np.uint8))
        finally:
            writer.close()
        log.info(f"Saved {self.record_n_videos} rollout video(s) to {path}")
