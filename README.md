# Nocturne Risk Research
#### Link to Original Nocturne Code
https://github.com/facebookresearch/nocturne

All code in the nocturne directory of this repo is sourced from the original code and modified to be operational on the cluster at WWU.

## Setup
```
cd /cluster/research-groups/ramasubramanian/workspace/$USER/
bash ./Miniconda3-latest-Linux-x86_64.sh -b -p /cluster/research-groups/ramasubramanian/workspace/$USER/miniconda3
source ~/.bashrc
git clone --depth 1 --shallow-submodules --recurse-submodules https://github.com/facebookresearch/nocturne.git
cd nocturne
conda env create -f environment.yml
conda activate nocturne
pip cache purge
conda install -y -c conda-forge sfml
conda install -y cmake util-linux gcc
ln -s /lib64/libudev.so.1 /cluster/research-groups/ramasubramanian/workspace/$USER/miniconda3/envs/nocturne/lib/libudev.so.0
python setup.py develop
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/
conda clean --all
```

## Future Work
#### Link to Fork of Nocturne Maintained by Author
https://github.com/Emerge-Lab/nocturne_lab
