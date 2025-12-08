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
echo "SLURM job started"
pwd

echo "Loading anaconda..."
module load anaconda

echo "Initializing conda..."
source /curc/sw/anaconda3/etc/profile.d/conda.sh

echo "Activating environment..."
conda activate podcast

echo "Starting Ollama..."
export OLLAMA_HOST=127.0.0.1:9999
nohup ollama serve > ollama_server.log 2>&1 &

echo "Waiting for Ollama..."
for i in {1..20}; do
    if curl -s http://127.0.0.1:9999/api/tags >/dev/null; then
        echo "Ollama is ready"
        break
    fi
    echo "Still waiting ($i)..."
    sleep 5
done

echo "Running python..."
python3 -c "print('Python works')"

python llama_question_type.py

echo "DONE"