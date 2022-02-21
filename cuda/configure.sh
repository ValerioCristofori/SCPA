#!/bin/sh
./clean.sh

cmake -DCMAKE_CUDA_ARCHITECTURES=75 -DBLOCK_SIZE=$1 .

cd ./lib
make
cd ..

make

