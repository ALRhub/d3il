from agents.models.robomimic.algo.algo import register_algo_factory_func, algo_name_to_factory_func, algo_factory, Algo, PolicyAlgo, ValueAlgo, PlannerAlgo, HierarchicalAlgo, RolloutPolicy

# note: these imports are needed to register these classes in the global algo registry
from agents.models.robomimic.algo.bc import BC, BC_Gaussian, BC_GMM, BC_VAE, BC_RNN, BC_RNN_GMM
from agents.models.robomimic.algo.bcq import BCQ, BCQ_GMM, BCQ_Distributional
from agents.models.robomimic.algo.cql import CQL
from agents.models.robomimic.algo.iql import IQL
from agents.models.robomimic.algo.gl import GL, GL_VAE, ValuePlanner
from agents.models.robomimic.algo.hbc import HBC
from agents.models.robomimic.algo.iris import IRIS
from agents.models.robomimic.algo.td3_bc import TD3_BC
