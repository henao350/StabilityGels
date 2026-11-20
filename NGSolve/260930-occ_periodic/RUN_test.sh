#!/bin/bash
#SBATCH --job-name=gelsMu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=36
#SBATCH --nodes=1
#SBATCH --nodelist=toki-hm-2
#### nodos con arquitectura compatible:     toki-hm-1,toki-hm-2,toki-hm-3,SRV03,tokikura,v100,dungu,wuruwe,SRV03,tokikura,v100,dungu,wuruwe

# imagen singularity pull docker://ngsxfem/ngsolve:latest

# Default values
CONTAINER="$HOME/ngsolve_latest.sif"
PRINTOUT_FILE="test.out"
singularity exec \
$CONTAINER \
python3 test_mesh.py \
2>&1 | tee -a $PRINTOUT_FILE

