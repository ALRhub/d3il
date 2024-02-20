MUJOCO_GL=egl python run_vision.py --config-name=stacking_vision_config \
              --multirun seed=0,1,2,3,4,5 \
              agents=bc_vision_agent \
              agent_name=bc_vision \
              window_size=1 \
              group=stacking_4_bc_seeds