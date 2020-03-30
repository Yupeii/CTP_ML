'''

Calc mismatch
2019-12-16
by Yupei

'''
import SimpleITK as sitk
import numpy as np
import os 

threshold = 6
results_path = './results'

rbf_path = os.path.join(results_path,'maps/rbf_map.nii')
ttp_path = os.path.join(results_path,'maps/ttp_map.nii')

img_rbf = sitk.ReadImage(rbf_path)
arr_rbf = sitk.GetArrayFromImage(img_rbf)
print(arr_rbf.max(), arr_rbf.min())

img_ttp = sitk.ReadImage(ttp_path)
arr_ttp = sitk.GetArrayFromImage(img_ttp)
print(arr_ttp.max(), arr_ttp.min())

(height, width, length) = arr_ttp.shape

# mismap = np.zeros(arr_ttp.shape)
map1 = np.zeros(arr_rbf.shape)
map2 = np.zeros(arr_ttp.shape)


thresh = arr_rbf.max()*0.3
# print(thresh)
flag_1 = arr_rbf < thresh
map1[flag_1] = 1 

flag1 = arr_rbf <= 1 # Where values are low
map1[flag1] = 0  # All low values set to 1 

flag2 = arr_ttp > threshold # Where values are low 
map2[flag2] = 1  # All low values set to 0 

# print(type(flag2))
# print(flag2.shape)
# print(flag2)
vol_rbf = 0
vol_tmax = 0  
for z in range(height):
	for x in range(width):
		for y in range(length):
			if flag1[z, x, y] == True:
				# print(z, x, y)
				vol_rbf+=1
			if flag2[z, x, y] == True:
				# print(z, x, y)
				vol_tmax+=1
print(vol_rbf)				
print(vol_tmax)



diff = vol_rbf - vol_tmax
ratio = vol_rbf/vol_tmax 
print('Diff: {}'.format(diff))
print('Ratio: {:.2f}'.format(ratio))

mismap = map1-map2
mismap = sitk.GetImageFromArray(mismap) 
map1 = sitk.GetImageFromArray(map1)
map2 = sitk.GetImageFromArray(map2)


# sitk.WriteImage(mismap, os.path.join(results_path,'mismatch/rbf_mismap.nii')) 
sitk.WriteImage(map1, os.path.join(results_path,'mismatch/map1.nii'))
sitk.WriteImage(map2, os.path.join(results_path,'mismatch/map2.nii'))
sitk.WriteImage(mismap, os.path.join(results_path,'mismatch/mismap.nii'))




