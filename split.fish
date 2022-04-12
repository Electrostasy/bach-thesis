#!/usr/bin/env fish

set -l options (fish_opt -s i -l input --required-val)
set options $options (fish_opt -s t -l training --required-val)
set options $options (fish_opt -s w -l wordlist --required-val)
argparse $options -- $argv or exit 1

# iconv -f (file -b --mime-encoding $_flag_i) -o "$_flag_i.utf8"
# iconv -f ISO_8859-1 -t UTF-8 -o "$_flag_i.utf8"

set -l length (wc -l $_flag_i | cut -d' ' -f1)
head -n (math -s 0 "$length * $_flag_t") $_flag_i > 'training.txt'
tail -n (math -s 0 "$length * $_flag_w") $_flag_i > 'wordlist.txt'
