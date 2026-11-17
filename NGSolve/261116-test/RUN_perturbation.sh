#!/bin/bash
#SBATCH --job-name=gelsMu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --nodes=1
#SBATCH --nodelist=toki-hm-1
#SBATCH --mem=4G
#### nodos con arquitectura compatible:     toki-hm-1,toki-hm-2,toki-hm-3,SRV03,tokikura,v100,dungu,wuruwe

# imagen singularity pull docker://ngsxfem/ngsolve:latest

# Default values
CONTAINER="$HOME/gels/ngsolve_latest.sif"
SIMULATION_NUMBER=9     # number of row to execute in the file to_execute.txt

PRINTOUT_FILE="gelsMuPerturbation$SIMULATION_NUMBER.out"

vars=$(awk "NR==$SIMULATION_NUMBER" to_execute.txt)
echo -e "Parámetros: $vars" | tee -a $PRINTOUT_FILE

singularity exec \
$CONTAINER \
python3 gels2dMuAddPerturbation.py $vars \
2>&1 | tee -a $PRINTOUT_FILE

echo -e "Screen printout available also at: slurm-$SLURM_JOB_ID.out\n###\n" | tee -a $PRINTOUT_FILE
