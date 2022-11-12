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
# PATH_TO_SAVE="${dataset}/pop${pop_size}_gen${gen_size}"
# mkdir -p $PATH_TO_SAVE

# python main.py $dataset $pop_size $gen_size $batch_size

# dataset="cifar10"
# pop_size=100
# gen_size=100
# batch_size=4
# PATH_TO_SAVE="${dataset}/pop${pop_size}_gen${gen_size}_run${i}"
# mkdir -p "results/$PATH_TO_SAVE"


# python main.py $dataset $pop_size $gen_size $batch_size $PATH_TO_SAVE
if [ -z "$1" ] && [ -z "$2" ]
then
    start=1
    end=10
else
    start=$1
    end=$2
fi

for (( i=$start; i<=$end; i++ ))
    dataset="MNIST"
    pop_size=1
    gen_size=1
    batch_size=4
    PATH_TO_SAVE="${dataset}/pop${pop_size}_gen${gen_size}_run${i}"
    mkdir -p "results/$PATH_TO_SAVE"
  

    python main.py $dataset $pop_size $gen_size $batch_size $PATH_TO_SAVE
    echo "run ${dataset} ${i} finished" >> run_script/running_info_${start}_${end}
done

# for (( i=$start; i<=$end; i++ ))
# do
#     dataset="cifar10"
#     pop_size=100
#     gen_size=100
#     batch_size=4
#     PATH_TO_SAVE="${dataset}/pop${pop_size}_gen${gen_size}_run${i}"
#     mkdir -p "results/$PATH_TO_SAVE"
 

#     python main.py $dataset $pop_size $gen_size $batch_size  $PATH_TO_SAVE

#     echo "run ${dataset} ${i} finished" >> run_script/running_info_${start}_${end}
# done

rm run_script/running_info_$1_$2
exit