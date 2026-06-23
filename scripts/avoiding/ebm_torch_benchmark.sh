python run.py --config-name=avoiding_config \
              --multirun seed=0,1,2,3,4,5 \
              agents=ebm_torch_agent \
              agent_name=ebm_torch \
              window_size=8 \
              group=avoiding_ebm_torch_seeds \
              simulation.n_cores=30 simulation.n_trajectories=480
