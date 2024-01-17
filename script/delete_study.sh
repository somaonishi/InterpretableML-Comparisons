#!/bin/bash

cd "$(dirname "$0")"
cd ../
source .venv/bin/activate
data=$1

echo $data

for model in rerx dt j48graft fbts rulecosi; do
    python main.py exp=optuna exp.delete_study=true exp.study_name=${data}-${model}-\${seed} seed='range(0,10)' -m
done
