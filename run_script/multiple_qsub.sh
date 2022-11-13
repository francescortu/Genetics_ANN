#!/bin/bash

#if the queqe is empty, it's better to parallelise the multiple runs in order to use more GPUs at the same time

qsub run_orfeo.sh 1 3
qsub run_orfeo.sh 4 6
qsub run_orfeo.sh 7 10