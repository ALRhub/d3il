from agents.models.robomimic.config.config import Config
from agents.models.robomimic.config.base_config import config_factory, get_all_registered_configs

# note: these imports are needed to register these classes in the global config registry
from agents.models.robomimic.config.bc_config import BCConfig
from agents.models.robomimic.config.bcq_config import BCQConfig
from agents.models.robomimic.config.cql_config import CQLConfig
from agents.models.robomimic.config.iql_config import IQLConfig
from agents.models.robomimic.config.gl_config import GLConfig
from agents.models.robomimic.config.hbc_config import HBCConfig
from agents.models.robomimic.config.iris_config import IRISConfig
from agents.models.robomimic.config.td3_bc_config import TD3_BCConfig