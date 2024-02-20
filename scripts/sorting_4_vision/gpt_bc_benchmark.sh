MUJOCO_GL=egl python run_vision.py --config-name=sorting_4_vision_config \
              --multirun seed=0,1,2,3,4,5 \
              agents=gpt_vision_agent \
              agent_name=gpt_bc \
              window_size=5 \
              group=sorting_4_gpt_bc_seeds \
              train_simulation.n_cores=10 \
              simulation.n_cores=10 \
              simulation.n_contexts=60 \
              simulation.n_trajectories_per_context=18