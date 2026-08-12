#!/bin/sh
#SBATCH --job-name=modeII
#SBATCH --ntasks=8
#SBATCH --partition=bb
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8

# Default values
NUM_PROCESSES=8     # that it be equal to the number of tasks specified in the sbatch directive --ntasks=...
MEF90_FOLDER="~"
CONTAINER="$MEF90_FOLDER/mef90ubuntu_0.3.0.sif"

# Usage help message
usage() {
    echo "Usage: $0 -g <GMSH_NAME> -n <SIMULATION_NAME>"
    echo "  -g  Specify the geometry name   (Required)"
    echo "  -n  Specify the simulation name (Required)"
    exit 1
}

# Parse command-line options
while getopts "g:n:" opt; do
    case "${opt}" in
        g) GMSH_NAME="${OPTARG}"; echo "GMSH_NAME=${GMSH_NAME}" ;;
        n) SIMULATION_NAME="${OPTARG}"; echo "SIMULATION_NAME=${SIMULATION_NAME}" ;;
        *) usage ;;
    esac
done

# Check if NAME was provided
if [ -z "$SIMULATION_NAME" ] || [ -z "$GMSH_NAME" ]; then
    echo "Error: Both SIMULATION_NAME and GMSH_NAME are required options."
    usage
fi

# Set variables based on SIMULATION_NAME#
LARGE_FILES_FOLDER="scratchpad.nosync"
#
GMSH_SCRIPT="${GMSH_NAME}.geo"
MESH="$LARGE_FILES_FOLDER/${GMSH_NAME}.msh"
#
RESULT="$LARGE_FILES_FOLDER/${SIMULATION_NAME}.exo"
OPTIONS_FILE="${SIMULATION_NAME}.yaml"

# delete $SIMULATION_NAME.out
rm $SIMULATION_NAME.out

# Run simulation with vDef
CMD_interior_clause=$(echo "mpirun -np $NUM_PROCESSES \
vDef -geometry $MESH \
-result $RESULT \
-options_file $OPTIONS_FILE")
CMD="apptainer exec -e \
$CONTAINER \
sh -c \"$CMD_interior_clause\" \
2>&1 | tee -a $SIMULATION_NAME.out"

echo "$CMD" | tee $SIMULATION_NAME.out
echo "Screen printout available also at: slurm-$SLURM_JOB_ID.out" | tee -a $SIMULATION_NAME.out
echo "----------------------------------------"
START_TIME=$(date "+%Y-%m-%d %H:%M:%S")
START_SECONDS=$(date +%s)
echo "Job started at: $START_TIME" | tee -a $SIMULATION_NAME.out
echo "----------------------------------------" | tee -a $SIMULATION_NAME.out
# # Generate mesh with Gmsh
# gmsh -2 "$GMSH_SCRIPT" -o "$MESH" 2>&1 | tee -a $SIMULATION_NAME.out
# Executes vDef
eval "$CMD"
unset CMD
END_TIME=$(date "+%Y-%m-%d %H:%M:%S")
END_SECONDS=$(date +%s)
ELAPSED_SECONDS=$((END_SECONDS - START_SECONDS))
ELAPSED_FORMAT=$(printf '%02d:%02d:%02d\n' $((ELAPSED_SECONDS/3600)) $((ELAPSED_SECONDS%3600/60)) $((ELAPSED_SECONDS%60)))
echo "Job finished at: $END_TIME" | tee -a $SIMULATION_NAME.out
echo "Total execution time: $ELAPSED_FORMAT ($ELAPSED_SECONDS seconds)" | tee -a $SIMULATION_NAME.out
