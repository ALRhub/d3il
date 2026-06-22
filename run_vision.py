import os
import logging
import random

# Silence gym's noisy import-time "upgrade to Gymnasium" notice (a raw print to
# stderr keyed by version in gym_notices; not a warnings.warn, so it can't be
# filtered). Must run before `import gym`.
try:
    import gym_notices.notices as _gym_notices
    _gym_notices.notices = {}
except Exception:
    pass

import hydra
import numpy as np
from tqdm import tqdm

import wandb
from omegaconf import DictConfig, OmegaConf
import torch


log = logging.getLogger(__name__)


OmegaConf.register_new_resolver(
     "add", lambda *numbers: sum(numbers)
)
torch.cuda.empty_cache()


def set_seed_everywhere(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


@hydra.main(config_path="configs", config_name="aligning_vision_config.yaml")
def main(cfg: DictConfig) -> None:

    # if cfg.seed in [0, 1]:
    #     os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    # elif cfg.seed in [2, 3]:
    #     os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    # elif cfg.seed in [4, 5]:
    #     os.environ["CUDA_VISIBLE_DEVICES"] = "2"

    set_seed_everywhere(cfg.seed)

    # init wandb logger and config from hydra path
    # Tags: method (agent_name, from <name>_benchmark.sh) / modality / task / seed
    from hydra.core.hydra_config import HydraConfig
    config_name = HydraConfig.get().job.config_name          # e.g. "aligning_vision_config"
    task = config_name.replace("_vision_config", "").replace("_config", "")
    modality = "image"                                       # run_vision.py drives the image-based benchmarks
    tags = [str(cfg.agent_name), f"task:{task}", f"modality:{modality}", f"seed:{cfg.seed}"]

    wandb_config = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=False)
    # Author placeholders in the repo configs -> fall back to your own entity.
    entity = OmegaConf.select(cfg, "wandb.entity", default=None)
    if entity in (None, "", "???", "davidjia", "atalaydonat"):
        entity = os.environ.get("WANDB_ENTITY")  # None => your default wandb entity
    project = os.environ.get("WANDB_PROJECT") or OmegaConf.select(cfg, "wandb.project", default=None)
    if project in (None, "", "???"):
        project = "d3il"
    mode = os.environ.get("WANDB_MODE") or OmegaConf.select(cfg, "wandb.mode", default="online")

    wandb.config = wandb_config
    run = wandb.init(
        project=project,
        entity=entity,
        group=cfg.group,
        name=f"{cfg.agent_name}_{task}",
        tags=tags,
        mode=mode,
        config=wandb_config,
    )

    agent = hydra.utils.instantiate(cfg.agents)

    best_success = -1
    train_sim = hydra.utils.instantiate(cfg.train_simulation)
    for num_epoch in tqdm(range(agent.epoch)):

        # train the agent
        agent.train_vision_agent()

        if not (num_epoch + 1) % agent.eval_every_n_epochs:

            successrate, _ = train_sim.test_agent(agent)

            if successrate > best_success:
                best_success = successrate

                agent.store_model_weights(agent.working_dir, sv_name=agent.eval_model_name)
                log.info('New best success rate. Stored weights have been updated!')

    agent.store_model_weights(agent.working_dir, sv_name=agent.last_model_name)
    log.info("Training done!")

    # load the model performs best on the evaluation set
    agent.load_pretrained_model(agent.working_dir, sv_name=agent.eval_model_name)

    # simulate the model
    env_sim = hydra.utils.instantiate(cfg.simulation)
    env_sim.test_agent(agent)

    log.info("done")

    wandb.finish()


if __name__ == "__main__":
    main()
