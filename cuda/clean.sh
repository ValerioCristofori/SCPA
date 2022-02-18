#!/bin/sh
rm Makefile
rm *.cmake
rm -R CMakeFiles
rm -R CMakeCache.txt
rm main

cd lib
./clean.sh
cd ..
