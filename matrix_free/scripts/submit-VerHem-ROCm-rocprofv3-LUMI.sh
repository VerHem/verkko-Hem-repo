#!/bin/bash -l
#SBATCH --job-name=VerHemG   # Job name
#SBATCH --output=VerHemG.o%j # Name of stdout output file
#SBATCH --error=VerHemG.e%j  # Name of stderr error file
#SBATCH --partition=dev-g  # partition name
#SBATCH --nodes=2               # Total number of nodes 
#SBATCH --ntasks-per-node=8     # 8 MPI ranks per node, 16 total (2x8)
#SBATCH --gpus-per-node=8       # Allocate one gpu per MPI rank
#SBATCH --time=0-00:30:00       # Run time (d-hh:mm:ss)
#SBATCH --account=project_462001601  # Project for billing

cat << EOF > select_gpu
#!/bin/bash

export MPICH_GPU_SUPPORT_ENABLED=1
export ROCR_VISIBLE_DEVICES=\$SLURM_LOCALID
if [ \$SLURM_PROCID -eq 0 ]; then 
  rocprofv3 --kokkos-trace --sys-trace --output-format=pftrace -- /projappl/project_462001601/VerHem/matrix_free/build/verhem
else
  /projappl/project_462001601/VerHem/matrix_free/build/verhem
fi
EOF

chmod +x ./select_gpu

loadmodule1601G

CPU_BIND="map_cpu:49,57,17,25,1,9,33,41"


# srun --cpu-bind=${CPU_BIND} ./select_gpu /projappl/project_462001601/VerHem/matrix_free/build/verhem
# srun rocprofv3 --kokkos-trace --sys-trace --output-format=pftrace -- /projappl/project_462001601/VerHem/matrix_free/build/verhem
srun --cpu-bind=${CPU_BIND} ./select_gpu

rm -rf ./select_gpu
