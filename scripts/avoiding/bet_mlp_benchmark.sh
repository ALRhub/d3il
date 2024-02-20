python run.py --config-name=avoiding_config \
              --multirun seed=0,1,2,3,4,5 \
              agents=bet_mlp_agent \
              agent_name=bet_mlp \
              window_size=1 \
              group=avoiding_bet_mlp_seeds \
              agents.model.vocab_size=64 \
              agents.model.offset_loss_scale=1.0