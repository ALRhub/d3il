MUJOCO_GL=egl python run_vision.py --config-name=aligning_vit_config \
              --multirun seed=0 \
              agents=dinov2_bc_vision_agent \
              agent_name=dinov2_bc_vision \
              window_size=1 \
              group=aligning_bc_seeds