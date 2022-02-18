#!/bin/bash

cd cuda
./configure.sh
cd ..

cd openmp
make 
cd ..
