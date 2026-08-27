#!/bin/bash
#SBATCH -n 3
#SBATCH -w gpu02
#SBATCH --gres=gpu:1

# added by Miniconda3 4.5.12 installer
# >>> conda init >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$(CONDA_REPORT_ERRORS=false '/gpu01/miniconda3/bin/conda' shell.bash hook 2> /dev/null)"
if [ $? -eq 0 ]; then
    \eval "$__conda_setup"
else
    if [ -f "/gpu01/miniconda3/etc/profile.d/conda.sh" ]; then
        . "/gpu01/miniconda3/etc/profile.d/conda.sh"
        CONDA_CHANGEPS1=false conda activate base
    else
        \export PATH="/gpu01/miniconda3/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda init <<<

#export CUDNN_PATH='/usr/local/cuda-11.4/lib64/libcudnn.so.7'
#export PATH=/usr/local/cuda-11.4/bin:$PATH
#export LD_LIBRARY_PATH=/usr/local/cuda-11.4/lib64:$LD_LIBRARY_PATH
#export LD_LIBRARY_PATH=/gpu01/cuda/lib64:$LD_LIBRARY_PATH
#export LD_LIBRARY_PATH="/gpu01/cuda/lib64"

conda activate tf1.13
date

cd $SLURM_SUBMIT_DIR
source /opt/rh/devtoolset-11/enable
ROOT=./
export PYTHONPATH=$PYTHONPATH:$ROOT
python LISTA.py