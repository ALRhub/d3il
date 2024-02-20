MUJOCO_GL=egl python run_vision.py --config-name=stacking_vision_config \
              --multirun seed=0,1,2,3,4,5 \
              agents=bet_mlp_vision_agent \
              agent_name=bet_mlp_vision \
              window_size=5 \
              group=stacking_bet_mlp_seeds
