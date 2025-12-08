#!/bin/bash

#SBATCH --account=blanca-blast-lecs
#SBATCH --qos=blanca-blast-lecs
#SBATCH --partition=blanca-blast-lecs

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:h100_3g.40gb:1
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=slurm_%j.out

#SBATCH --mail-user=niklas.hofstetter@colorado.edu
#SBATCH --mail-type=ALL
#SBATCH --job-name=llama_questions

source ~/.bashrc

echo "SLURM job started"
pwd

echo "Loading anaconda..."
module load anaconda

echo "Activating conda..."
conda activate podcast

echo "Running python..."
python3 -c "print('Python works')"

echo "Starting server"
export OLLAMA_HOST=127.0.0.1:9999
nohup ollama serve > log.txt 2>&1 &
sleep 10  # Give the server some time to start

python llama_question_type.py

echo "DONE"