#!/bin/bash

#SBATCH --job-name=gemm
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32


./dgemm
