#!/bin/bash

dirarr=($(echo t-0.{50..95..5}))
rTarr=($(echo 0.{50..95..5}))

NumElem=${#rTarr[@]}

for ((n=0;n<NumElem;n++)); do

    cd ${dirarr[$n]}

    pwd

    sbatch submit-3.sh

    sleep 30s
    
    cd ..

done    
