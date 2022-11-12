#!/bin/bash
#PBS -l walltime=24:00:00
#PBS -q dssc_gpu
#PBS -l nodes=1:ppn=1

cd $PBS_O_WORKDIR
cd ..
module load conda
conda activate deep_le
touch run_script/running_info
# dataset="MNIST"
# pop_size=100
# gen_size=100
# batch_size=4
# PATH_TO_SAVE="results/${dataset}/pop${pop_size}_gen${gen_size}"
# mkdir -p $PATH_TO_SAVE

# python main.py $dataset $pop_size $gen_size $batch_size

dataset="cifar10"
pop_size=100
gen_size=100
batch_size=4
PATH_TO_SAVE="results/${dataset}/pop${pop_size}_gen${gen_size}"
mkdir -p $PATH_TO_SAVE

python main.py $dataset $pop_size $gen_size $batch_size

for i in {1..10}
do
    dataset="MNIST"
    pop_size=100
    gen_size=100
    batch_size=4
    PATH_TO_SAVE="results/${dataset}/pop${pop_size}_gen${gen_size}_run${i}"
    mkdir -p $PATH_TO_SAVE

    python main.py $dataset $pop_size $gen_size $batch_size
    echo "run ${dataset} ${i} finished" >> run_script/running_info
done

for i in {1..10}
do
    dataset="cifar10"
    pop_size=100
    gen_size=100
    batch_size=4
    PATH_TO_SAVE="results/${dataset}/pop${pop_size}_gen${gen_size}_run${i}"
    mkdir -p $PATH_TO_SAVE

    python main.py $dataset $pop_size $gen_size $batch_size

    echo "run ${dataset} ${i} finished" >> run_script/running_info
done

rm run_script/running_info
exit