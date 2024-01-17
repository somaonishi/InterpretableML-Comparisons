#!/bin/bash

cd "$(dirname "$0")"

data=$1

echo $data

./rerx.sh $data
./rulecosi.sh $data
./fbts.sh $data
./j48graft.sh $data
./dt.sh $data
