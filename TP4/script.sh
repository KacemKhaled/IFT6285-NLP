#!/bin/bash

for size in 100 200 300; do
  for window in 3 4 5 6 7; do
    for negative in 4 5 6; do
      log_file="size"$size"_window"$window"_negative$negative"
      echo "-----------------------------------------------------------------------"
      echo "Training --size $size --window $window --negative $negative > "$log_file".log"
      { 
        time python TP4.py --size $size --window $window --negative $negative  2> logs/"$log_file".log
      } 2> logs/time_"$log_file".log
      echo "-----------------------------------------------------------------------"
    done
  done
done
python plot_metaparams_impacts.py