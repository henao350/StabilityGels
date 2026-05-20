CMD="rsync -avzP --exclude='*.sif' phernandez@172.16.105.194:~/260519-Cluster-test-ChemicalPotential .."
echo "Command being executed:"
echo "$CMD"
echo " "
eval $CMD