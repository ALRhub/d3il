python run.py --config-name=avoiding_config \
              --multirun seed=0,1,2,3,4,5 \
              agents=gpt_bc_agent \
              agent_name=gpt_bc \
              window_size=5 \
              group=avoiding_gpt_bc_seeds