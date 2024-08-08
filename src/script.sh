#!/bin/sh
nb_teachers="10"
for nb_tchrs in $nb_teachers
do
    echo ">>> Number of teachers : " $nb_tchrs
    for nb_fair in $(seq 3 $nb_tchrs)
    do
        python3 fairness_accuracy_impact_eval.py $nb_tchrs $nb_fair
    done
done