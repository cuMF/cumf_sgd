#! /bin/bash


make clean
make

mv netflix_train.txt netflix_train
mv netflix_test.txt netflix_test

./transform netflix_train
./transform netflix_test
