# D3IL_Benchmark

This project encompasses the D3IL Benchmark, comprising 7 robot learning tasks: Avoiding, Pushing,
Aligning, Sorting, Stacking, Inserting, and Arranging. All these environments are implemented 
using Mujoco and Gym. The [D3IL](environments/d3il) directory includes the robot controller along with the environment 
implementations, while the [Agents](agents) directory provides 11 imitation learning methods encompassing 
both state-based and image-based policies.

<p align="center">

  <img width="100.0%" src="figures/github_readme.gif">

</p>

## Installation
```
# assuming you already have conda installed
bash install.sh
```

## Usage

### File System

```
D3IL_Benchmark
└── agents # model implementation
    └── models
    ...
└── configs # task configs and model hyper parameters
└── environments
    └── d3il    
        └── d3il_sim    # code for controller, robot, camera etc.
        └── envs        # gym environments for all tasks
        └── models      # object xml files
        ...
    └── dataset # data saving folder and data process
        └── data
        ...
└── scripts # running scripts and hyper parameters
    └── aligning
    └── stacking
    ...
└── simulation # task simulation
...
```
### Download the dataset

Donwload the zip file and extract the data into the folder `environments/dataset/`

```
cd environments/dataset/
wget https://mega.nz/file/ZXkQjYxS#728-v_DDliocn1U84cpEtkHHw42XiX8L4vmvEA7OrQ0
unzip -d data/
```

### Reproduce the results

We conducted extensive experiments for imitation learning methods, spanning deterministic policies to 
multi-modal policies, and from MLP-based models to Transformer-based models. To reproduce the results 
mentioned in the paper, use the following commands:

Train state-based MLP on the Pushing task
```
bash scripts/aligning/bc_benchmark.sh
```

Train state-based BeT on the Aligning task
```
bash scripts/aligning/bet_benchmark.sh
```

Train image-based DDPM-ACT on the sorting task
```
bash scripts/sorting_4_vision/ddpm_encdec_benchmark.sh
```

### Train your models
We offer a unified interface for integrating new algorithms:

- Add your method in `agents/models/`
- Read `agents/base_agent.py` and `agents/bc_agent.py` and implement your new agent there
- Add your agent config file in `configs/agents/`
- Add a training scripts in `scripts/aligning/`

### Creating Custom Tasks
Our simulation system, built on Mujoco and Gym, allows the creation of new tasks. In order to create new tasks, please 
refer to the [D3il_Guide](environments/d3il/README.md)

After creating your task and recording data, simulate imitation learning methods on your task by following these steps:

- Read `environments/dataset/base_dataset.py` and `environments/dataset/pushing_dataset.py` and 
implement your task dataset there
- Read `configs/pushing_config.yaml` and Add your task config file in `configs/`
- Read `simulation/base_sim.py` and `simulation/pushing_sim.py` and implement 
your task simulation there

### Recording your own data
We provide the script `environments/d3il/gamepad_control/record_data.py` to record data for any task using a gamepad controller.
To record data for the tasks we provided, run `record_data.py -t <task>`. If you made a custom task, you need to add it to the script. Data that you record will be saved in the folder `environments/d3il/gamepad_control/data/<task>/recorded_data/`. The controls are as follows:

- `Right stick` to move the robot
- `A` to save the current episode
- `Y` to drop the current episode, reset the environment and start recording
- `B` to stop recording (but continue the episode)
- `A` to start recording

Please note that when `record_data.py` is first called, it starts recording by default.

## Key Components
- We use [Wandb](https://wandb.ai/site) to manage the experiments, so you should **specify your wandb account and project** in each task config file.
- We split the models into MLP-based and history-based methods; adjust `window_size` for different methods accordingly

### Acknowledgements

The code of this repository relies on the following existing codebases:

- BeT agent adapted from [bet](https://github.com/notmahi/bet).
- ACT agent from [act](https://github.com/tonyzhaozh/act)
- Diffusion Policy from [diffusion_policy](https://github.com/real-stanford/diffusion_policy)
- Beso Agent from [beso](https://github.com/intuitive-robots/beso)
- Implicit Behavior Cloning (IBC) Agent is inspired by [Kevin Zakka's reimplementation in torch](https://github.com/kevinzakka/ibc) 