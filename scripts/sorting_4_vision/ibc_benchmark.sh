MUJOCO_GL=egl python run_vision.py --config-name=sorting_4_vision_config \
              --multirun seed=0,1,2,3,4,5 \
              agents=ibc_vision_agent \
              agent_name=ibc_vision \
              window_size=1 \
              group=sorting_4_ibc_seeds \
              agents.sampler.sampler_stepsize_init=0.0493