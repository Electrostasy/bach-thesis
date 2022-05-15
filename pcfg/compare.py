#!/usr/bin/env python

from sys import argv
from getopt import getopt
from typing import TextIO


def compare(left: TextIO, right: TextIO):
    left_lines = {line for line in left.readlines()}
    right_lines = {line for line in right.readlines()}
    return left_lines.intersection(right_lines)


if __name__ == '__main__':
    left_path = ''
    right_path = ''
    dump_path = ''
    opts, _ = getopt(argv[1:], shortopts='hl:r:d:')
    for option, value in opts:
        match option:
            case '-h':
                # compare_against_hashed()
                pass
            case '-l':
                left_path = value
            case '-r':
                right_path = value
            case '-d':
                dump_path = value

    if left_path != '' and right_path != '':
        with open(left_path, 'r') as left, open(right_path, 'r') as right:
            intersection = compare(left, right)
            print(f"Matches: {len(intersection)}")

            if dump_path != '':
                with open(dump_path, 'w') as dump:
                    dump.writelines(intersection)
