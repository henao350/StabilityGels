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

singularity exec \
$CONTAINER \
python3 generate_mesh_periodic.py 
