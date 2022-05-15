#!/usr/bin/env fish

set -l options (fish_opt -s w -l wordlist --required-val)
set options $options (fish_opt -s m -l model --required-val)
argparse $options -- $argv or exit 1

set -l counts (for p in (seq 1 9); math "pow(10,$p)"; end)

function trim_ex --argument-names 'filename'
  echo (string split -r -m1 . $filename)[1]
end

set -l wordlist $_flag_w
set -l model $_flag_m

for count in $counts
  set -l guesses "pcfg_$(trim_ex $wordlist)_$(trim_ex $model)_$count.txt"
  ./pcfg.py -l $count -w $wordlist -f $model -s $guesses
end
