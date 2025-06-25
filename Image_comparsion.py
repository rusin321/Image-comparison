# -*- coding: utf8 -*-

import matplotlib.pyplot as plt
import numpy as np
import os
import subprocess
import time
import shutil
from PIL import Image
import torch
import cv2
from piq import ssim, psnr

def func(start, end):
	start = start + np.clip(-(i + start), a_min = None, a_max = 0)
	end = end + np.clip(Blu_Ray_directory_len - 1 - (i + end), a_min = None, a_max = 0)
	
	for j in [start, end]:
		Blu_Ray = cv2.imread(os.path.join(Blu_Ray_directory, Blu_Ray_images[Blu_Ray_index+j]))
		Blu_Ray = torch.from_numpy(np.array([np.transpose(Blu_Ray)])).cuda()
	
		if j < 0:
			ssim_list.insert(0, ssim(Blu_Ray, DVD, data_range = 255).cpu().numpy().item() )
			j_list.insert(0, j)
			Blu_ray_list.insert(0, os.path.join(Blu_Ray_directory, Blu_Ray_images[Blu_Ray_index+j]))
		elif j > 0:
			ssim_list.append( ssim(Blu_Ray, DVD, data_range = 255).cpu().numpy().item() )
			j_list.append(j)
			Blu_ray_list.append(os.path.join(Blu_Ray_directory, Blu_Ray_images[Blu_Ray_index+j]))
	
	if max(ssim_list) < 0.9 and len(j_list) <= 31:
		func(start-1, end+1)


#Opening images
Blu_Ray_directory = r"F:\ATLA\Original datasets\1x_ATLA_Blu-Ray_deblur\S02E15\Part 3\Blu-Ray"
DVD_directory = r"F:\ATLA\Original datasets\1x_ATLA_Blu-Ray_deblur\S02E15\Part 3\DVD"

DVD_directory_HQ = r"F:\ATLA\Original datasets\1x_ATLA_Blu-Ray_deblur\S02E15\Part 3\HQ"
Blu_Ray_directory_LQ = r"F:\ATLA\Original datasets\1x_ATLA_Blu-Ray_deblur\S02E15\Part 3\LQ"

#Comparing images
start = -1
end = 2

Blu_Ray_directory_len = len(os.listdir(Blu_Ray_directory))
DVD_directory_len = len(os.listdir(DVD_directory))

DVD_images = os.listdir(DVD_directory)
Blu_Ray_images = os.listdir(Blu_Ray_directory)

ssim_big_list = []

for i in range(DVD_directory_len):
	ssim_list = []
	DVD_list = []
	Blu_ray_list = []
	j_list = []
	
	#Opening images from Blu-Ray
	DVD = cv2.imread(os.path.join(DVD_directory, DVD_images[i]))
	DVD = torch.from_numpy(np.array([np.transpose(DVD)])).cuda()
	
	#Determining indices
	Blu_Ray_index = round(i*0.8)
	
	if Blu_Ray_index > Blu_Ray_directory_len-1:
		Blu_Ray_index = Blu_Ray_directory_len-1
	
	if i == 0:
		start = 0
		end = 2
	elif i > 0 and i < (DVD_directory_len-1):
		start = -1
		end = 2
	elif i == (DVD_directory_len-1):
		start = -1
		end = 1
	
	#Comparing Blu-Raya with DVD
	for j in range(start, end):
		Blu_Ray = cv2.imread(os.path.join(Blu_Ray_directory, Blu_Ray_images[Blu_Ray_index+j]))
		Blu_Ray = torch.from_numpy(np.array([np.transpose(Blu_Ray)])).cuda()
		
		ssim_list.append( ssim(Blu_Ray, DVD, data_range = 255).cpu().numpy().item() )
		j_list.append(j)
		Blu_ray_list.append(os.path.join(Blu_Ray_directory, Blu_Ray_images[Blu_Ray_index+j]))
	
	if max(ssim_list) < 0.9: #ssim: 0.90, psnr: 27
		func(start-1, end)
	
	if len(j_list) > 3:
		print("j =", j_list[ ssim_list.index(max(ssim_list)) ])
		print("len(j) =", len(j_list))
		print(max(ssim_list))
	
	ssim_big_list.append(max(ssim_list))
	
	if ssim_big_list[i] >= 0.79:
		shutil.copy( os.path.join(DVD_directory, DVD_images[i]), DVD_directory_HQ + "\\{}.png".format( str(i+1).zfill(5) ) )
		
		shutil.copy( Blu_ray_list[ ssim_list.index(max(ssim_list)) ], Blu_Ray_directory_LQ + "\\{}.png".format( str(i+1).zfill(5) ) )
	
	print(str(int(Blu_ray_list[ ssim_list.index(max(ssim_list)) ].split(".")[0].split("\\")[-1])) + " --> " + str(i+1))


os.system("pause")
