#!/bin/bash
#SBATCH --job-name=gelsMu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=36
#SBATCH --nodes=1
#SBATCH --nodelist=toki-hm-1
#### nodos con arquitectura compatible:     toki-hm-1,toki-hm-2,toki-hm-3,SRV03,tokikura,v100,dungu,wuruwe,toki-hm-1,toki-hm-2,toki-hm-3,SRV03,tokikura,v100,dungu,wuruwe


# imagen singularity pull docker://ngsxfem/ngsolve:latest

# Default values
CONTAINER="$HOME/ngsolve_latest.sif"

vars=$(awk "NR==1" to_execute.txt)
echo "Parámetros: $vars"

PRINTOUT_FILE="gelsMu1.out"

singularity exec \
$CONTAINER \
python3 gels2d_ChemicalPotential.py $vars \
2>&1 | tee -a $PRINTOUT_FILE

echo "Screen printout available also at: slurm-$SLURM_JOB_ID.out" | tee -a $PRINTOUT_FILE
