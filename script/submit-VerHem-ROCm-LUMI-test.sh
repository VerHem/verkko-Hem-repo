#!/bin/bash -l
#SBATCH --job-name=VerHemG   # Job name
#SBATCH --output=VerHemG.o%j # Name of stdout output file
#SBATCH --error=VerHemG.e%j  # Name of stderr error file
#SBATCH --partition=dev-g  # partition name
#SBATCH --nodes=1               # Total number of nodes 
#SBATCH --ntasks-per-node=8     # 8 MPI ranks per node, 16 total (2x8)
#SBATCH --gpus-per-node=8       # Allocate one gpu per MPI rank
#SBATCH --time=0-00:20:00       # Run time (d-hh:mm:ss)
#SBATCH --account=project_462001601  # Project for billing

cat << EOF > select_gpu
#!/bin/bash

export ROCR_VISIBLE_DEVICES=\$SLURM_LOCALID
exec \$*
EOF

chmod +x ./select_gpu

CPU_BIND="map_cpu:49,57,17,25,1,9,33,41"

export MPICH_GPU_SUPPORT_ENABLED=1

srun --cpu-bind=${CPU_BIND} ./select_gpu /projappl/project_462001601/VerHem/matrix_free/build/verhem
rm -rf ./select_gpu
