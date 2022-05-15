#!/usr/bin/env fish

cat $argv[1] | while read -l line; echo $line | iconv -f utf-8 -t us-ascii//TRANSLIT; end > "$(echo (string split -r -m1 . $argv[1])[1])-ascii.txt"
