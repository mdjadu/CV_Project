#!/usr/bin/python3

# N should be a positive integer. 

import argparse
import numpy as np

def crop_array(arr_2d, offset_height, offset_width, target_height,target_width):
	arr_2d_crop = arr_2d[offset_height:offset_height+target_height,offset_width:offset_width+target_width]
	return arr_2d_crop


def row_mul(n,p):
	A = np.eye(n)
	P = np.vstack((A[0::2], A[1::2]))
	P_crop = crop_array(P,p[0],p[1],p[2],p[3])
	row_padd = np.ones((1,P_crop.shape[1])) * 0.5
	P_padd = np.vstack((row_padd,P_crop,row_padd))
	col_padd = np.ones((P_crop.shape[0]+2,1)) * 0.5
	P_padd = np.hstack((col_padd,P_padd,col_padd))
	P_concat = np.hstack((P_padd,P_padd))
	return P, P_crop, P_padd, P_concat


parser = argparse.ArgumentParser()
parser.add_argument("--N", type=int, help="order of the matrix")
parser.add_argument("--crop", nargs='+', type=int, help="Dimensions for cropping (list of 4 integers)")
args = parser.parse_args()

if args.N <= 0:
	print("N should be a positive integer.")
elif args.crop:
	P, P_crop, P_padd, P_concat = row_mul(args.N,args.crop)
	print("Original array:\n{0}\n\nCropped array:\n{1}\n\nPadded array:\n{2}\n\nConcatenated array: shape = {3}\n{4}".format(P, P_crop, P_padd, P_concat.shape, P_concat))
else:
	P, P_crop, P_padd, P_concat = row_mul(args.N,[1,1,args.N-2,args.N-2])
	print("Original array:\n{0}\n\nCropped array:\n{1}\n\nPadded array:\n{2}\n\nConcatenated array: shape = {3}\n{4}".format(P, P_crop, P_padd, P_concat.shape, P_concat))	
