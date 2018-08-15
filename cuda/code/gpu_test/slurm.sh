#!/bin/bash
#SBATCH -J "my_gpu_job"
#SBATCH -n 1
#SBATCH -p gpu_q
#SBATCH -t 00:30:00
srun -n1  ./a.out
