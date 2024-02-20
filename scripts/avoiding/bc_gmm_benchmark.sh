#python run.py --config-name=avoiding_config \
#              --multirun seed=42 \
#              agents=bc_gmm_agent \
#              agent_name=bc_gmm \
#              window_size=1 \
#              group=avoiding_bc_gmm_sweep \
#              agents.model.n_gaussians=8,16,24,32,64

python run.py --config-name=avoiding_config \
              --multirun seed=0,1,2,3,4,5 \
              agents=bc_gmm_agent \
              agent_name=bc_gmm \
              window_size=1 \
              group=avoiding_bc_gmm_seeds \
              agents.model.n_gaussians=8