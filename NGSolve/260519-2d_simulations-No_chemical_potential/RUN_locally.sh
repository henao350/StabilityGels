#!/bin/bash
# vars=$(awk "NR==$SLURM_ARRAY_TASK_ID" to_execute.txt)
vars=" 90 1.62 3 14 15"
echo "Parámetros: $vars"

# singularity exec ~/ngsolve_latest.sif python3 gels-testCluster-ChemicalPotential4.py $vars
python3 copy.release_rate.py $vars
