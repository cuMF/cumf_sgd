#! /bin/bash


make clean
make

mv netflix_mm.txt netflix_mm
mv netflix_mme.txt netflix_mme

./transform netflix_mm
./transform netflix_mme