#!/bin/bash

MAT="data/matrix/*"
vector="data/vector/vector.txt"
THREADS=4

for mat in $MAT
do
# FAILSAFE #
# Check if "$f" FILE exists and is a regular file and then only copy it #
  if [ -f "$mat" ]
  then
	./main -serial $mat $vector 
	./main -ompCSR $mat $vector $THREADS
	./main -ompELLPACK $mat $vector $THREADS
  else
    echo "Warning: Some problem with \"$mat\""
  fi
done
