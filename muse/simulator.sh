#!/bin/bash

cnt=5
outdir=simu_output

function init_env() {
  if [ -d $outdir ]; then
    cd $outdir
    rm -rf *
    cd ..
  else
    mkdir -p $outdir
  fi

  if [ -n "$1" ]; then
    cnt=$1
  fi

  if [ "$1" == "clean" ]; then
    rm -rf $outdir
    cd muse-v3
    python3 main.py clean
    cd ..
  fi
}

function simulator() {
  for ((i=1; i<=cnt; i++))
  do
    cd muse-v3 
    printf 'Start muse-v3 toolchains simulator %d times...\n' $i
    python3 main.py debug > a
    sub_dir=../$outdir/output_$i
    mv output $sub_dir
    cp input/simulator.py $sub_dir
    mv .debug/simulator.log $sub_dir
    printf 'Save %s to dir successfully.\n' $outdir/outdir_$i
    cd ..
  done
}

init_env $@
simulator

