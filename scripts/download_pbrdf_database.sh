#!/bin/bash

while IFS= read -r line
do
    wget http://vclab.kaist.ac.kr/siggraph2020/pbrdfdataset/"$line"_matlab.zip -P $1
    unzip $1/"$line"_matlab.zip -d $1/"$line"_matlab
done < $2
