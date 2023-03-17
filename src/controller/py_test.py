import argparse
import sys
import numpy as np
import ast


def add(a, b):
    return len(a) + len(b)


def print_arr(a):
    for i, row in enumerate(a):
        if i % 10 == 0:
            print(row)


parser = argparse.ArgumentParser()
parser.add_argument('--position', type=str)
parser.add_argument('--velocity', type=str)

args = parser.parse_args(sys.argv)
print(add(ast.literal_eval(args.cloth_position), ast.literal_eval(args.cloth_velocity)))
