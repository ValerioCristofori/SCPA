#!/bin/bash

CUDA_DIR="cuda"
MAT="../data/matrix/*"
vector="../data/vector/vector.txt"


cd $CUDA_DIR
pwd

for i in 128 256 384 512 640 768 896 1024;
do
    ./configure.sh $i

    for mat in $MAT
    do
    # FAILSAFE #
    # Check if "$f" FILE exists and is a regular file and then only copy it #
      if [ -f "$mat" ]
      then
      echo -e "---------- Calculate \"$mat\" --------------\n" 

        ./main -cudaCSR $mat $vector  
        ./main -cudaELLPACK $mat $vector 

      echo -e "--------------------------------------------\n\n" 

      else
        echo "Warning: Some problem with \"$mat\""
      fi
    done
done

cd ..