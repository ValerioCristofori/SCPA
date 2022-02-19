#!/bin/bash

LOG_FILE="calc.log"
CUDA_DIR="./cuda"
MAT="data/matrix/*"
vector="data/vector/vector.txt"

for mat in $MAT
do
# FAILSAFE #
# Check if "$f" FILE exists and is a regular file and then only copy it #
  if [ -f "$mat" ]
  then
  echo -e "---------- Calculate \"$mat\" --------------\n" >> $LOG_FILE  

  $CUDA_DIR/main -cudaCSR $mat $vector >> $LOG_FILE 
  $CUDA_DIR/main -cudaELLPACK $mat $vector >> $LOG_FILE 

  echo -e "--------------------------------------------\n\n" >> $LOG_FILE  

  else
    echo "Warning: Some problem with \"$mat\""
  fi
done
