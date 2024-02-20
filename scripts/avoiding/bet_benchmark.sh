python run.py --config-name=avoiding_config \
              --multirun seed=0,1,2,3,4,5 \
              agents=bet_agent \
              agent_name=bet \
              window_size=5 \
              group=avoiding_bet_seeds \
              agents.model.vocab_size=64 \
              agents.model.offset_loss_scale=1.0