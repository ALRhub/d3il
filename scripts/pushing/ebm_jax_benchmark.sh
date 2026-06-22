python run.py --config-name=pushing_config \
              --multirun seed=0,1,2,3,4,5 \
              agents=ebm_jax_agent \
              agent_name=ebm_jax \
              window_size=1 \
              group=pushing_ebm_jax_seeds
