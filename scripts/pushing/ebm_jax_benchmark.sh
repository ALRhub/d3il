python run.py --config-name=pushing_config \
              --multirun seed=0,1,2,3,4,5 \
              agents=ebm_jax_agent \
              agent_name=ebm_jax \
              window_size=8 \
              group=pushing_ebm_jax_seeds \
              simulation.n_cores=30 simulation.n_contexts=30 simulation.n_trajectories_per_context=16
