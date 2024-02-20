python run.py --config-name=avoiding_config \
              --multirun seed=0,1,2,3,4,5 \
              agents=act_agent \
              agent_name=act \
              window_size=3 \
              group=avoiding_act_seeds \
              agents.kl_loss_factor=0.01