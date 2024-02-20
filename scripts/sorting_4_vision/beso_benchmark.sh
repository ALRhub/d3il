MUJOCO_GL=egl python run_vision.py --config-name=sorting_4_vision_config \
              --multirun seed=0,1,2,3,4,5 \
              agents=beso_vision_agent \
              agent_name=beso_vision \
              window_size=5 \
              group=sorting_4_beso_seeds \
              agents.num_sampling_steps=16 \
              agents.sigma_min=0.1 \
              agents.sigma_max=1
