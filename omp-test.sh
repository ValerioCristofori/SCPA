#!/bin/bash

LOG_FILE="calc.log"
OMP_DIR="./openmp"
#MAT="/data/vcristofori/data/matrix/*"
#vector="/data/vcristofori/data/vector/vector.txt"
MAT="./data/matrix/*"
vector="./data/vector/vector.txt"

cores=$(grep -c ^processor /proc/cpuinfo) 

for mat in $MAT
do
# FAILSAFE #
# Check if "$f" FILE exists and is a regular file and then only copy it #
  if [ -f "$mat" ]
  then
  echo -e "---------- Calculate \"$mat\" --------------\n"

  $OMP_DIR/main -serial $mat $vector

  for ((i=1; i<=cores; i++)); do
      $OMP_DIR/main -ompCSR $mat $vector $i 
      $OMP_DIR/main -ompELLPACK $mat $vector $i
  done

  echo -e "--------------------------------------------\n\n" 

  else
    echo "Warning: Some problem with \"$mat\""
  fi
done
