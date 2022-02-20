#!/bin/bash

LOG_FILE="calc.log"
OMP_DIR="./openmp"
MAT="/data/vcristofori/data/matrix/*"
vector="/data/vcristofori/data/vector/vector.txt"

for mat in $MAT
do
# FAILSAFE #
# Check if "$f" FILE exists and is a regular file and then only copy it #
  if [ -f "$mat" ]
  then
  echo -e "---------- Calculate \"$mat\" --------------\n"

	$OMP_DIR/main -serial $mat $vector 
	
  $OMP_DIR/main -ompCSR $mat $vector 4 
  $OMP_DIR/main -ompELLPACK $mat $vector 4
  $OMP_DIR/main -ompCSR $mat $vector 8
  $OMP_DIR/main -ompELLPACK $mat $vector 8
   $OMP_DIR/main -ompCSR $mat $vector 12
  $OMP_DIR/main -ompELLPACK $mat $vector 12
  $OMP_DIR/main -ompCSR $mat $vector 16
  $OMP_DIR/main -ompELLPACK $mat $vector 16
$OMP_DIR/main -ompCSR $mat $vector 20 
  $OMP_DIR/main -ompELLPACK $mat $vector 20
$OMP_DIR/main -ompCSR $mat $vector 24 
  $OMP_DIR/main -ompELLPACK $mat $vector 24
$OMP_DIR/main -ompCSR $mat $vector 28 
  $OMP_DIR/main -ompELLPACK $mat $vector 28 
  $OMP_DIR/main -ompCSR $mat $vector 32 
  $OMP_DIR/main -ompELLPACK $mat $vector 32 
  $OMP_DIR/main -ompCSR $mat $vector 36 
  $OMP_DIR/main -ompELLPACK $mat $vector 36  



  echo -e "--------------------------------------------\n\n" 

  else
    echo "Warning: Some problem with \"$mat\""
  fi
done
