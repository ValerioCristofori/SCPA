#!/bin/bash

LOG_FILE="calc.log"
OMP_DIR="./openmp"
MAT="data/matrix/*"
vector="data/vector/vector.txt"

for mat in $MAT
do
# FAILSAFE #
# Check if "$f" FILE exists and is a regular file and then only copy it #
  if [ -f "$mat" ]
  then
  echo -e "---------- Calculate \"$mat\" --------------\n" >> $LOG_FILE  

	$OMP_DIR/main -serial $mat $vector >> $LOG_FILE 
	
  $OMP_DIR/main -ompCSR $mat $vector 4 >> $LOG_FILE 
	$OMP_DIR/main -ompELLPACK $mat $vector 4 >> $LOG_FILE 

  $OMP_DIR/main -ompCSR $mat $vector 8 >> $LOG_FILE 
  $OMP_DIR/main -ompELLPACK $mat $vector 8 >> $LOG_FILE 

  $OMP_DIR/main -ompCSR $mat $vector 16 >> $LOG_FILE 
  $OMP_DIR/main -ompELLPACK $mat $vector 16 >> $LOG_FILE 

  echo -e "--------------------------------------------\n\n" >> $LOG_FILE  

  else
    echo "Warning: Some problem with \"$mat\""
  fi
done
