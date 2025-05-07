#!/bin/bash

dirarr=($(echo t-0.{50..95..5}))
rTarr=($(echo 0.{50..95..5}))

NumElem=${#rTarr[@]}

for ii in ${dirarr[@]}; do

    mkdir $ii && cd $ii

    pwd

    cp /scratch/project_462000604/test-after-lumi-update/test-dealii-9.5-Trilinos-14.4-VII-breakgoodjob-0.96MDOF/*.sh .
    cp /scratch/project_462000604/test-after-lumi-update/test-dealii-9.5-Trilinos-14.4-VII-breakgoodjob-0.96MDOF/configuration.prm .

    cd ..

done

for ((n=0;n<NumElem;n++)); do

    cd ${dirarr[$n]}

    pwd

    sed -i "s/set pressure in bar  = 32.0/set pressure in bar  = 28.0/g" configuration.prm
    sed -i "s/set t_reduced        = 0.83/set t_reduced        = ${rTarr[${n}]}/g" configuration.prm
    # sed -i "s/seed                1/seed                 61/g" 
    # sed -i "s/IniT                 1.6687/IniT                 2.8536/g" 
    # sed -i "s/Inip                26.0/Inip                6.0/g" 
    # sed -i "s/InitH               0.0, 0.0, 0.0/InitH               0.0, 0.0, 150.0/g" 
    # sed -i "s/do_gapA_clip         yes/do_gapA_clip         no/g" 
    # sed -i "s/BCs1                    periodic/BCs1                    PairBreaking/g" 
    # sed -i "s/BCs2                    periodic/BCs2                    PairBreaking/g" 

    # less sim_config_dyGiLa-Langevin*.txt
    sed -i "s/test/SM/g" submit-3.sh
    sed -i "s/partition=debug/partition=standard/g" submit-3.sh
    sed -i "s/time=0-00:30:00/time=0-01:30:00/g" submit-3.sh
    cd ..

done    
