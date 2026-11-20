#!/bin/bash
cd "$HOME/mnt/short-horizon-challenge/hft500"; C=$HOME/scratch/cache_valid; P=$HOME/hftenv/bin/python
for i in BTCUSDT_PERP ETHUSDT_PERP SOLUSDT_PERP 1000PEPEUSDT_PERP; do
  $P scripts/stage9_build.py $i --period valid --cache-dir $C --max-seconds 140 >> reports/stage12_cache_$i.log 2>&1 &
done; wait
for i in $C/*; do echo -n "$(basename $i)=$(ls $i/*.parquet 2>/dev/null | wc -l) "; done; echo
