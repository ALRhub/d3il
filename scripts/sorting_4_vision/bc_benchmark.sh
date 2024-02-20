MUJOCO_GL=egl python run_vision.py --config-name=sorting_4_vision_config \
              --multirun seed=0,1,2,3,4,5 \
              agents=bc_vision_agent \
              agent_name=bc_vision \
              window_size=1 \
              simulation.n_cores=30 \
              simulation.n_contexts=60 \
              simulation.n_trajectories_per_context=18 \
              group=sorting_4_bc_seeds