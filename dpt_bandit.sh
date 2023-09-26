#!/bin/bash

#SBATCH -p psych_gpu
#SBATCH -c 36
#SBATCH -G 4
#SBATCH -t 1-00:00:00
#SBATCH --mem=372g

module load miniconda CUDAcore cuDNN
conda init bash
conda activate gofar

#insert directory of DPT_bandit.py below
cd
python -u DPT_bandit.py
