#!/bin/bash
#SBATCH --job-name=gelsMu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
# nodes whose cpu arquitecture is compatible with singularity ngsolve: 
# toki-hm-1,toki-hm-2,toki-hm-3,SRV03,tokikura,v100,dungu,wuruwe
#SBATCH --nodelist=toki-hm-1,toki-hm-2,toki-hm-3
#SBATCH --array=1-9

# imagen singularity pull docker://ngsxfem/ngsolve:latest

# vars=$(awk "NR==$SLURM_ARRAY_TASK_ID" to_execute.txt)
vars=$(awk "NR==9" to_execute.txt)
echo "Parámetros: $vars"

singularity exec ~/ngsolve_latest.sif python3 gels-testCluster-ChemicalPotential4.py $vars
