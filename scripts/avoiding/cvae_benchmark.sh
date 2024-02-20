python run.py --config-name=avoiding_config \
              --multirun seed=0,1,2,3,4,5 \
              agents=cvae_agent \
              agent_name=cvae \
              window_size=1 \
              group=avoiding_cvae_seeds \
              agents.model.encoder.latent_dim=32 \
              agents.kl_loss_factor=59.874124563262455