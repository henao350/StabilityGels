#!/bin/bash
#SBATCH --job-name=gelsNoMu
#SBATCH --ntasks=1 
#SBATCH --cpus-per-task=16
#SBATCH --array=1-1

# imagen singularity pull docker://ngsxfem/ngsolve:latest

vars=$(awk "NR==$SLURM_ARRAY_TASK_ID" to_execute.txt)
echo $vars
singularity exec ultrasound.sif python3 gels-testCluster-NoChemicalPotential.py $vars
