#!/bin/bash -e

#SBATCH --nodes=1
#SBATCH --time=00:30:00
#SBATCH --gres=gpu:v100:2
#SBATCH --job-name=send-recv

apptainer run docker://joelberkeley/send-recv  # add NVIDIA support ... https://apptainer.org/docs/user/main/gpu.html
