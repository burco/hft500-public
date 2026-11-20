#!/bin/bash
# one ~165 s slice of the Stage 9 build: one worker per instrument, resumable
cd "$HOME/mnt/short-horizon-challenge/hft500"; C=$HOME/scratch/cache; P=$HOME/hftenv/bin/python
for i in BTCUSDT_PERP ETHUSDT_PERP SOLUSDT_PERP 1000PEPEUSDT_PERP; do
  $P scripts/stage9_build.py $i --cache-dir $C --max-seconds 145 >> reports/stage9_$i.build.log 2>&1 &
done; wait
for i in $C/*; do echo -n "$(basename $i)=$(ls $i/*.parquet | wc -l) "; done; echo
