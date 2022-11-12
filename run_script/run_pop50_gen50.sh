#!/bin/bash
#PBS -l walltime=09:00:00
#PBS -q dssc_gpu
#PBS -l nodes=1:ppn=1

cd $PBS_O_WORKDIR
cd ..
module load conda
conda activate deep_le
python main.py 