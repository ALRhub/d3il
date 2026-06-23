python run.py --config-name=stacking_config \
              --multirun seed=0,1,2,3,4,5 \
              agents=ebm_torch_mlp_agent \
              agent_name=ebm_torch_mlp \
              window_size=8 \
              group=stacking_ebm_torch_mlp_seeds \
              simulation.n_cores=30 simulation.n_contexts=60 simulation.n_trajectories_per_context=18
