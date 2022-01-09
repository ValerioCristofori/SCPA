#!/bin/sh
rm Makefile
rm *.cmake
rm -R CMakeFiles
rm -R CMakeCache.txt
rm output.txt cuda-CSR cuda-ELLPACK

cd libraries
./clean.sh
cd ..
