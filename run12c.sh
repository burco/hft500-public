#!/bin/bash
# usage: run12c.sh DATE T1 T2 ...   (official-style truncation test of the submission, subprocess runs)
D=$1; shift; DD=$HOME/scratch/ddir_$D; S="$HOME/mnt/short-horizon-challenge"; PY=$HOME/scratch/subvenv/bin/python; V="$HOME/hftenv/bin/python $S/hft500/scripts/stage12_verify.py"
[ -f $HOME/scratch/pred_final_$D.parquet ] || (cd $S/submission && $PY predict.py --data-dir $DD --date $D --out $HOME/scratch/pred_final_$D.parquet >/dev/null)
for T in "$@"; do
  rm -rf $HOME/scratch/tdir; $V truncate --src $DD --T $T --dest $HOME/scratch/tdir >/dev/null
  (cd $S/submission && $PY predict.py --data-dir $HOME/scratch/tdir --date $D --out $HOME/scratch/pred_trunc.parquet >/dev/null)
  echo "$D $T $($V compare --full $HOME/scratch/pred_final_$D.parquet --trunc $HOME/scratch/pred_trunc.parquet --T $T)" | tee -a $S/hft500/reports/stage12_causality.log
done
