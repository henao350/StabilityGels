#!/bin/bash
SIMULATION_NUMBER=2     # number of row to execute in the file to_execute.txt

vars=$(awk "NR==$SIMULATION_NUMBER" to_execute.txt)
echo "Parámetros: $vars"

PRINTOUT_FILE="gelsMu$SIMULATION_NUMBER.out"

python3 gels2dMu.py $vars \
2>&1 | tee -a $PRINTOUT_FILE
