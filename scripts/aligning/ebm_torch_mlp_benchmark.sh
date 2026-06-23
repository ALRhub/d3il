python run.py --config-name=aligning_config \
              --multirun seed=0,1,2,3,4,5 \
              agents=ebm_torch_mlp_agent \
              agent_name=ebm_torch_mlp \
              window_size=10 \
              group=aligning_ebm_torch_mlp_seeds
