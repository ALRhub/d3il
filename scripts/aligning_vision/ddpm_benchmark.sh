MUJOCO_GL=egl python run_vision.py --config-name=aligning_vision_config \
              --multirun seed=0,1,2,3,4,5 \
              agents=ddpm_vision_agent \
              agent_name=ddpm_vision \
              window_size=1 \
              group=aligning_ddpm_seeds \
              agents.model.model.model.t_dim=4 \
              agents.model.model.n_timesteps=4
