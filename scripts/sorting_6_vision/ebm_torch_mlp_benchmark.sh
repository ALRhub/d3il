MUJOCO_GL=egl python run_vision.py --config-name=sorting_6_vision_config \
              --multirun seed=0,1,2,3,4,5 \
              agents=ebm_torch_mlp_vision_agent \
              agent_name=ebm_torch_mlp_vision \
              window_size=8 \
              group=sorting_6_vision_ebm_torch_mlp_seeds
