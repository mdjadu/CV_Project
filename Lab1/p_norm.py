#!/usr/bin/python3

# p should be positive real number. 
# n should be a vector either positive or negative real number(s).

import argparse
import math

def norm_p(n,p):
	accumulate = 0
	for x in n:
		x = abs(x)
		accumulate += x**p
	norm = math.exp(math.log(accumulate)/p)
	return norm

parser = argparse.ArgumentParser()
parser.add_argument("n", type=float, nargs='+', help="input vector")
parser.add_argument("--p", type=float, help="order of the norm")
args = parser.parse_args()
if args.p:
	if args.p <= 0:
		print("p should be a positive real number.")
	else:
		print("Norm of {0} is {1:.2f}".format(args.n, norm_p(args.n, args.p)))
else:
	print("Norm of {0} is {1:.2f}".format(args.n, norm_p(args.n, 2)))
