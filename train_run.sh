#!/bin/bash

#SBATCH --account=blanca-blast-lecs
#SBATCH --qos=blanca-blast-lecs
#SBATCH --partition=blanca-blast-lecs

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:h100:1   
#SBATCH --mem=4G
#SBATCH --time=24:00:00
#SBATCH --output=slurm_%j.out

#SBATCH --mail-user=niklas.hofstetter@colorado.edu
#SBATCH --mail-type=ALL
#SBATCH --job-name=get_data

source ~/.bashrc

echo "SLURM job started"
pwd

echo "Loading anaconda..."
module load anaconda

echo "Activating conda..."
conda activate podcast

echo "Running python..."
python3 -c "print('Python works')"
python train.py

echo "DONE"