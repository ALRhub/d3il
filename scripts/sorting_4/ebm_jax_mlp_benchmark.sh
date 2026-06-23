python run.py --config-name=sorting_4_config \
              --multirun seed=0,1,2,3,4,5 \
              agents=ebm_jax_mlp_agent \
              agent_name=ebm_jax_mlp \
              window_size=8 \
              group=sorting_4_ebm_jax_mlp_seeds \
              simulation.n_cores=60 simulation.n_contexts=60 simulation.n_trajectories_per_context=18
