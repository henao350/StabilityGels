#!/bin/bash

# imagen singularity pull docker://ngsxfem/ngsolve:latest

vars=$(awk "NR==1" to_execute.txt)
echo "Parámetros: $vars"

# singularity exec ~/ngsolve_latest.sif python3 gels-testCluster-ChemicalPotential4.py $vars
python3 gels2d_ChemicalPotential.py $vars
