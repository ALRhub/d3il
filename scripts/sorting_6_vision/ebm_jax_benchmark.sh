MUJOCO_GL=egl python run_vision.py --config-name=sorting_6_vision_config \
              --multirun seed=0,1,2,3,4,5 \
              agents=ebm_jax_vision_agent \
              agent_name=ebm_jax_vision \
              window_size=8 \
              group=sorting_6_vision_ebm_jax_seeds
