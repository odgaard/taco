#!/bin/bash
#SBATCH -N 1
#SBATCH --exclusive

set -u

export SUITESPARSE_PATH=/home/rubensl/misc/data/suitesparse
source /home/${USER}/.bashrc
source activate hypermapper

sspath=/home/rubensl/misc/data/suitesparse
out=experiments

mkdir -p "$out"

for i in $sspath/*.mtx; do

	matrix="$(basename $i)"
	echo "$matrix"
	./bin/taco-taco_dse --count 0 --matrix_name "$matrix" --op SDDMM
	retVal=$?
	if [ $retVal -ne 0 ]; then 
		continue
	fi
	./bin/taco-taco_dse --count 1 --matrix_name "$matrix" --op SDDMM
	./bin/taco-taco_dse --count 2 --matrix_name "$matrix" --op SDDMM
	./bin/taco-taco_dse --count 3 --matrix_name "$matrix" --op SDDMM
	./bin/taco-taco_dse --count 4 --matrix_name "$matrix" --op SDDMM
	./bin/taco-taco_dse --count 5 --matrix_name "$matrix" --op SDDMM
	./bin/taco-taco_dse --count 6 --matrix_name "$matrix" --op SDDMM
	./bin/taco-taco_dse --count 7 --matrix_name "$matrix" --op SDDMM
	./bin/taco-taco_dse --count 8 --matrix_name "$matrix" --op SDDMM
	./bin/taco-taco_dse --count 9 --matrix_name "$matrix" --op SDDMM

	./bin/taco-taco_dse --count 0 --matrix_name "$matrix" --method random_sampling --op SDDMM
	./bin/taco-taco_dse --count 1 --matrix_name "$matrix" --method random_sampling --op SDDMM
	./bin/taco-taco_dse --count 2 --matrix_name "$matrix" --method random_sampling --op SDDMM
	./bin/taco-taco_dse --count 3 --matrix_name "$matrix" --method random_sampling --op SDDMM
	./bin/taco-taco_dse --count 4 --matrix_name "$matrix" --method random_sampling --op SDDMM
	./bin/taco-taco_dse --count 5 --matrix_name "$matrix" --method random_sampling --op SDDMM
	./bin/taco-taco_dse --count 6 --matrix_name "$matrix" --method random_sampling --op SDDMM
	./bin/taco-taco_dse --count 7 --matrix_name "$matrix" --method random_sampling --op SDDMM
	./bin/taco-taco_dse --count 8 --matrix_name "$matrix" --method random_sampling --op SDDMM
	./bin/taco-taco_dse --count 9 --matrix_name "$matrix" --method random_sampling --op SDDMM
done #<$1
