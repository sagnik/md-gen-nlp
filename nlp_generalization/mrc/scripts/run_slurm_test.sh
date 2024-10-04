#!/bin/bash
# The interpreter used to execute the script

#“#SBATCH” directives that convey submission options:

#SBATCH --job-name=evaluate-mrc-models
#SBATCH --mail-type=END
#SBATCH --nodes=1
#SBATCH --partition=gpu_mig40
#SBATCH --gpus=1
#SBATCH --mem-per-gpu=128000m
#SBATCH --cpus-per-gpu=1
#SBATCH --time=1:00:00
#SBATCH --account=vgvinodv0
#SBATCH --output=./slurms/slurm-%j.out
#SBATCH --error=./slurms/slurm-%j.err
# The application(s) to execute along with its input arguments and options:
module load python/3.10.4
poetry run python evaluate_finetuned_mrc_models.py --config $1
