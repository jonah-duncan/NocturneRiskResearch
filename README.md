# Nocturne Risk Research
#### Link to Original Nocturne Code
https://github.com/facebookresearch/nocturne

All code in the nocturne directory of this repo is sourced from the original code and modified to be operational on the cluster at WWU. This does not contain all code needed to run nocturne.

## Accessing Cluster via Jupyter
https://cluster.cs.wwu.edu/jupyter/index.html

## Primary Environment Setup
``` bash
# Download and install Miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
cd /cluster/research-groups/ramasubramanian/workspace/$USER/
bash ./Miniconda3-latest-Linux-x86_64.sh -b -p /cluster/research-groups/ramasubramanian/workspace/$USER/miniconda3
source ~/.bashrc
git clone --depth 1 --shallow-submodules --recurse-submodules https://github.com/facebookresearch/nocturne.git
cd nocturne
# Make sure to copy environment.yml / cfgs files from this repo to local nocturne folder
# ...
conda env create -f environment.yml
conda activate nocturne
pip cache purge
conda install -y -c conda-forge sfml
conda install -y cmake util-linux gcc
# Create a symbolic link to pretend we have an old version of udev
ln -s /lib64/libudev.so.1 /cluster/research-groups/ramasubramanian/workspace/$USER/miniconda3/envs/nocturne/lib/libudev.so.0
# Actually build the Nocturne code
python setup.py develop
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/
conda clean --all
```

## *N*th Environment Setup
``` bash
# Download and install Miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
git clone --depth 1 --shallow-submodules --recurse-submodules https://github.com/facebookresearch/nocturne.git nocturneN
# Make sure to copy environment.yml / cfgs files from this repo to local nocturne folder
# ...
grep -rl 'nocturne' "nocturneN/" | xargs sed -i 's/nocturne/nocturneN/g'
grep -rl 'nocturneN_mini' "nocturneN/" | xargs sed -i 's/nocturneN_mini/nocturne_mini/g'
cd nocturneN
conda env create -f environment.yml
conda activate nocturneN
pip cache purge
conda install -y -c conda-forge sfml
conda install -y cmake util-linux gcc
# Rename the follwing nocturneN
    /cluster/research-groups/ramasubramanian/workspace/$USER/nocturneN/nocturne/pybind11/include/nocturne.h
    /cluster/research-groups/ramasubramanian/workspace/$USER/nocturneN/nocturne/pybind11/src/nocturne.cc
    /cluster/research-groups/ramasubramanian/workspace/$USER/nocturneN/examples/on_policy_files/nocturne_runner.py
    /cluster/research-groups/ramasubramanian/workspace/$USER/nocturneN/examples/nocturne_functions.py
    /cluster/research-groups/ramasubramanian/workspace/$USER/nocturneN/nocturne/
# Create a symbolic link to pretend we have an old version of udev
ln -s /lib64/libudev.so.1 /cluster/research-groups/ramasubramanian/workspace/$USER/miniconda3/envs/nocturneN/lib/libudev.so.0
# Actually build the Nocturne code
python setup.py develop
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/
conda clean --all
```

## Assorted Notes
+ GPU 0: NVIDIA A100-PCIE-40GB (UUID: GPU-cb631598-15e4-a29e-6dae-c136a03eaafc)
+ Use _cuda_ instead of _cuda:0_
+ Completely remove nocturne environment by deleting the folder and running:
```
conda remove --name nocturne --all
```

## Sample Factory Fixes
+ Change _${device}_ to _cuda_ in _config.yaml_ and _APPO.yaml_
+ Remove the following from _/cluster/research-groups/ramasubramanian/workspace/$USER/miniconda3/envs/nocturne/lib/python3.8/site-packages/sample_factory/algorithms/utils/arguments.py_
```
for key, value in cfg.cli_args.items():
    if key in loaded_cfg and loaded_cfg[key] != value:
        log.debug('Overriding arg %r with value %r passed from command line', key, value)
        loaded_cfg[key] = value
```
+ Modify _appo_utils_ in the _/cluster/research-groups/ramasubramanian/workspace/$USER/miniconda3/envs/nocturne/lib/python3.8/site-packages/sample_factory/algorithms/appo/_ below to force gpu to be found
```
def get_available_gpus():
    #orig_visible_devices = os.environ[f'{CUDA_ENVVAR}_backup_']
    #available_gpus = [int(g) for g in orig_visible_devices.split(',') if g]
    available_gpus = 'GPU-cb631598-15e4-a29e-6dae-c136a03eaafc'    
    return available_gpus
```
+ Modify _learner.py_ in _/cluster/research-groups/ramasubramanian/workspace/$USER/miniconda3/envs/nocturne/lib/python3.8/site-packages/sample_factory/algorithms/appo/_ on line 461 to multiply float by float
```
# add max entropy to the rewards
if self.cfg.max_entropy_coeff != 0.0:
    with timing.add_time('max_entropy'), torch.no_grad():
        action_distr_params = buffer.action_logits.reshape((-1, buffer.action_logits.shape[-1]))  # [E*T, A]
        entropies = get_action_distribution(self.action_space, torch.Tensor(action_distr_params)).entropy().numpy()  # [E*T]
        entropies = entropies.reshape((-1, self.cfg.rollout))  # [E, T]
        #buffer.rewards += self.cfg.max_entropy_coeff * entropies  # [E, T]
        #buffer.rewards += float(self.cfg.max_entropy_coeff) * entropies  # [E, T]
        #print(self.cfg.max_entropy_coeff)
        s = self.cfg.max_entropy_coeff
        s = s.replace(',', '')
        buffer.rewards += float(s) * entropies  # [E, T]
```
+ Modify _learner.py_  in _/cluster/research-groups/ramasubramanian/workspace/$USER/miniconda3/envs/nocturne/lib/python3.8/site-packages/sample_factory/algorithms/appo/_ on line 661 to be:
```
tmp_filepath = join(checkpoint_dir, 'a.temp_checkpoint')
```
+ To visualize sample factory results a few things need to be done:
```
conda install -c conda-forge cudnn
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/python3.8/site-packages/nvidia/cudnn/lib
# Change the torch device to 'cuda' on line 163 of /cluster/research-groups/ramasubramanian/workspace/$USER/miniconda3/envs/nocturne/lib/python3.8/site-packages/torch/serialization.py
```

## Accomplished
#### Sample Factory
+ Got functional without errors/breaks
+ Can train and visualize models 
+ Trained models are not performant even after 15+ hours of training
#### Imitation Learning
+ I got _train.py_ working in the summer of 2023 but lost that configuration of nocturne
#### On Policy
+ I never got the example code to run without errors
#### Rllib
+ I never got the example code to run without errors
+ This shows a lot of promise since it has DQN agents built in

## Future Work
#### Link to Fork of Nocturne Maintained by Author
https://github.com/Emerge-Lab/nocturne_lab

#### Current Progress on Setup (Not Guaranteed to Work)
``` bash
git clone --depth 1 --shallow-submodules --recurse-submodules https://github.com/Emerge-Lab/nocturne_lab.git
cd nocturne_lab
conda env create -f ./environment.yml
#change nocturne to nocturne_lab in environment.yml
conda activate nocturne_lab

pip install poetry
# Synchronise and update git submodules
git submodule sync
git submodule update --init --recursive
poetry install

pip install torch==2.3.0
conda install -y -c conda-forge sfml
conda install -y cmake util-linux gcc
#conda install -y gcc_linux-64 gxx_linux-64
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/
ln -s /lib64/libudev.so.1 /cluster/research-groups/ramasubramanian/workspace/$USER/miniconda3/envs/nocturne_lab/lib/libudev.so.0
pip install moviepy==1.0.3 pybind11==2.11.1 python-box pyvirtualdisplay==3.0 stable-baselines3==2.1.0 typer==0.9.0 gym==0.26.2 wandb tensorboard xvfbwrapper
pip install --upgrade pyvirtualdisplay  
```
