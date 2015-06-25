#!/bin/bash

mkdir testdata traindata
echo '[CHECK] DIVIDING DATA'
python dividedata.py ${*:1}
export NNJSON=$4

echo '[CHECK] TRAINING DATA'
for file in $(ls traindata)
do
	cat traindata/$file | python trainmap.py | sort | python trainreduce.py
done

echo '[CHECK] EVOLVING NETS'
for file in $(ls testdata)
do
	cat testdata/$file | python testmap.py | sort | python testreduce.py
done

echo '[CHECK] DELETING DATA'
rm -rf testdata traindata
