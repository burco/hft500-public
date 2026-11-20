#!/bin/bash
cd "$HOME/mnt/short-horizon-challenge/hft500"
for w in 0 1 2; do $HOME/hftenv/bin/python scripts/stage10_final.py validate --worker $w --workers 3 --budget 115 >> reports/stage10_valid_w$w.log 2>&1 & done; wait
tail -qn1 reports/stage10_valid_w*.log
