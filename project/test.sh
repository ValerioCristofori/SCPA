#!/bin/bash

CUDA_DIR="./cuda"
OMP_DIR="./openmp"
MAT="data/matrix/*"
vector="data/vector/vector.txt"
THREADS=4

for mat in $MAT
do
# FAILSAFE #
# Check if "$f" FILE exists and is a regular file and then only copy it #
  if [ -f "$mat" ]
  then
	$OMP_DIR/main -serial $mat $vector 
	$OMP_DIR/main -ompCSR $mat $vector $THREADS
	$OMP_DIR/main -ompELLPACK $mat $vector $THREADS

  $CUDA_DIR/main -cudaCSR $mat $vector
  $CUDA_DIR/main -cudaELLPACK $mat $vector

  else
    echo "Warning: Some problem with \"$mat\""
  fi
done
