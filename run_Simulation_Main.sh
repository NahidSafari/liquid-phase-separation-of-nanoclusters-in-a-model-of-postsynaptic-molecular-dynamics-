#!/bin/bash
#SBatch -p medium
#SBATCH -t 48:00:00
#SBATCH -o job_output/job-%J.out
#SBATCH -C scratch
#SBATCH -n 1
#SBATCH -c 11
#SBATCH -a 0-10

module purge
module load anaconda3
source activate pyrid-env

python3 -u /......./FixedConcentration_PatchyModel.py $SLURM_ARRAY_TASK_ID

conda deactivate
