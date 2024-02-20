python run.py --config-name=avoiding_config \
              --multirun seed=0,1,2,3,4,5 \
              agents=ibc_agent \
              agent_name=ibc \
              window_size=1 \
              group=avoiding_ibc_seeds \
              agents.sampler.sampler_stepsize_init=0.0493