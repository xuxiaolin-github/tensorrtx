#!/bin/bash

cd build

rm -rf ./

cmake ..

make

./yolov5  -s  ../yolov5s.wts yolov5s.engine s

./yolov5  -d  yolov5.engine 0
