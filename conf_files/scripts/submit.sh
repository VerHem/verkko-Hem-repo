#!/bin/bash -l
#SBATCH --job-name=PDW-ri   # Job name
#SBATCH --output=PDW-ri.o%j # Name of stdout output file
#SBATCH --error=PDW-ri.e%j  # Name of stderr error file
#SBATCH --partition=standard   # partition name
#SBATCH --nodes=128               # Total number of nodes 
#SBATCH --ntasks=8192            # Total number of mpi tasks
#SBATCH --mem=0                 # Allocate all the memory on each node
#SBATCH --time=2-00:00:00       # Run time (d-hh:mm:ss)
#SBATCH --account=project_462000926  # Project for billing

# All commands must follow the #SBATCH directives

setupConf="setup_config";

refineCycle0="refine-cycle_0";
refineCycle1="refine-cycle_1";
refineCycle2="refine-cycle_2";
refineCycle3="refine-cycle_3";
refineCycle4="refine-cycle_4";

if [ -e $setupConf ] && [ -e $refineCycle0 ] && [ -e $refineCycle1 ] && [ -e $refineCycle2 ] && [ -e $refineCycle3 ] && [ -e $refineCycle4 ];
then
    echo " All Folders exist\n ";
else
    mkdir setup_config refine-cycle_{0,1,2,3,4};
fi    
    
# Launch MPI code 
srun /projappl/project_462000926/build-2dPDW-Random-ini-I/sol/VerHem  # Use srun instead of mpirun or mpiexec
