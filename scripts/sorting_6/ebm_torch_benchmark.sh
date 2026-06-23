python run.py --config-name=sorting_6_config \
              --multirun seed=0,1,2,3,4,5 \
              agents=ebm_torch_agent \
              agent_name=ebm_torch \
              window_size=8 \
              group=sorting_6_ebm_torch_seeds \
              simulation.n_cores=60 simulation.n_contexts=60 simulation.n_trajectories_per_context=20
