#!/bin/bash
#BATCH --job-name=info_gan_run
#SBATCH --output=info_gan.log
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --mem=128gb
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH -C V100
RUNPATH=/home/khickey/gans_scratch/infogan
cd $RUNPATH
source infogan_env/bin/activate
PYTHONIOENCODING=utf8 python3 main.py