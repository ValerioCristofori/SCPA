#!/bin/sh
./clean.sh

cmake -DCMAKE_CUDA_ARCHITECTURES=75 .

cd ./lib
make
cd ..

make

