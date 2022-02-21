#!/bin/bash

cd cuda
./configure.sh 256
cd ..

cd openmp
make 
cd ..
