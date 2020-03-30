'''

To generate the entire maps for four params with trained ML models:

2019-12-12 by Yupei 

'''

import numpy as np 
from sklearn import metrics 
from lupy import *
import SimpleITK as sitk 
from joblib import load
import argparse
import yaml


# load params in config
parser = argparse.ArgumentParser(description='Ensambling')
parser.add_argument('--config', default='config.yaml')
args = parser.parse_args()
with open(args.config) as f:
    config = yaml.load(f)
for k,v in config.items():
    setattr(args, k, v)

complied_arr = []# Final array to hold all imput&output data from all patients
arr_input = []
arr_ttp = []
arr_rbf = []
arr_rbv = []
arr_mtt = []
arr_tmax = []
arr_length = [] 


# data_path = '../../data/ISLES2018/TRAINING' #local address
data_path = args.data_path_local
results_path = args.results_path

files = fsearch(path = data_path, suffix = '.nii', include='4DPWI', sort_level=-3)
cbv_path = fsearch(path = data_path, suffix = '.nii', include='CBV', sort_level=-3)
cbf_path = fsearch(path = data_path, suffix = '.nii', include='CBF', sort_level=-3)
mtt_path = fsearch(path = data_path, suffix = '.nii', include='MTT', sort_level=-3)
tmax_path = fsearch(path = data_path, suffix = '.nii', include='Tmax', sort_level=-3)
numFiles = len(files) 


svc_mtt = load(os.path.join(results_path, 'models/mtt_auto.joblib'))
svc_ttp = load(os.path.join(results_path, 'models/ttp_auto.joblib'))
svc_rbv = load(os.path.join(results_path, 'models/rbv_auto.joblib'))
svc_rbf = load(os.path.join(results_path, 'models/rbf_auto.joblib'))
#To load all inputs and outputs
nfile = 0
for file in files:
	print(file.split('/')[-3])
	npoint = 0
	val_arr = []
	point_dict = {}

	img = sitk.ReadImage(file)
	img_arr = sitk.GetArrayFromImage(img) #(49, 8, 256, 256)
	if len(img_arr.shape) < 4:
		print('Error: Invalid input data shape!')
		continue

	# img_mask = img_arr
	[t, height, width, length] = np.shape(img_arr)
	# print(t, height, width, length)

	img_cbv = sitk.ReadImage(cbv_path[nfile])
	img_cbf = sitk.ReadImage(cbf_path[nfile])
	img_mtt = sitk.ReadImage(mtt_path[nfile])
	img_tmax = sitk.ReadImage(tmax_path[nfile])

	img_cbv_arr = sitk.GetArrayFromImage(img_cbv)
	img_cbf_arr = sitk.GetArrayFromImage(img_cbf) 
	img_mtt_arr = sitk.GetArrayFromImage(img_mtt) 
	img_tmax_arr = sitk.GetArrayFromImage(img_tmax) 

	img_mtt_arr = img_mtt_arr*10

	img_mtt_arr = np.ndarray.astype(img_mtt_arr, 'int')
	img_tmax_arr= np.ndarray.astype(img_tmax_arr, 'int')


	for z in range(height):
		for j in range(width):
			for i in range(length):
				if (npoint+1)%100000 ==0: 
					print('reading file {} point {}'.format(nfile+1, npoint+1))  

				x_loc = j
				y_loc = i
				sliceNum = z        
				# print(z, j, i)

				######################################################################## 


				# get perfusion data
				xp = []
				fp = []
				for time in range(t):
					xp.append(time)
					fp.append(img_arr[time, sliceNum, x_loc, y_loc])

				# interpolation
				x = list(np.arange(0, t, t/100))
				# print(len(x)) #100
				interp = np.interp(x, xp, fp) 

				arr_length.append(len(interp))

				# k = 0
				# while (k < 4):

				rBV_val = img_cbv_arr[sliceNum, x_loc, y_loc]
				rBF_val  = img_cbf_arr[sliceNum, x_loc, y_loc]
				MTT_val = img_mtt_arr[sliceNum, x_loc, y_loc]
				Tmax_val = img_tmax_arr[sliceNum, x_loc, y_loc]

				# k+=1

				#create an entry to hold input values (x,y,z) and output values (TTP, rBF, rBV, MTT,TMAX)
				#we can use this data and enter it to scikit.learn
				arr_input.append(interp)
				arr_tmax.append(Tmax_val)
				arr_rbf.append(rBF_val)
				arr_rbv.append(rBV_val)
				arr_mtt.append(MTT_val)
				npoint += 1
	nfile += 1 
	if nfile+1 > 1:
		break


min_length = min(arr_length)

index = arr_length.index(min_length)

tempi = [t[:min_length] for t in arr_input]

arr_input = tempi

print(len(arr_input))

#ALL DATA ARRAYS for all points
input_all = arr_input
output_ttp_all= arr_tmax
output_rbv_all = arr_rbv
output_rbf_all = arr_rbf
output_mtt_all = arr_mtt


######################################################################## 

#To predict all

rbf_pred_all = svc_rbf.predict(input_all)
rbv_pred_all = svc_rbv.predict(input_all)
mtt_pred_all = svc_mtt.predict(input_all)
ttp_pred_all = svc_ttp.predict(input_all)

print('SVM no params specified ')

rbv_acc = metrics.accuracy_score(output_rbv_all, rbv_pred_all.round())
rbf_acc = metrics.accuracy_score(output_rbf_all, rbf_pred_all.round())
mtt_acc = metrics.accuracy_score(output_mtt_all, mtt_pred_all.round())
ttp_acc = metrics.accuracy_score(output_ttp_all, ttp_pred_all.round())

print('\tAccuracy \t NRMSE:') 

ymax = max(output_rbv_all)
ymin = min(output_rbv_all)
mse = metrics.mean_squared_error(output_rbv_all, rbv_pred_all.round())
rmse = np.sqrt(mse)
nrmse_rbv = rmse/(ymax - ymin)
print('\nrBV: ', rbv_acc, nrmse_rbv)

ymax = max(output_rbf_all)
ymin = min(output_rbf_all)
mse = metrics.mean_squared_error(output_rbf_all, rbf_pred_all.round())
rmse = np.sqrt(mse)
nrmse_rbf = rmse/(ymax - ymin)
print('\nrBF: ', rbf_acc, nrmse_rbf)

ymax = max(output_mtt_all)
ymin = min(output_mtt_all)
mse = metrics.mean_squared_error(output_mtt_all, mtt_pred_all.round())
rmse = np.sqrt(mse)
nrmse_mtt = rmse/(ymax - ymin)
print('\nMTT: ', mtt_acc, nrmse_mtt)

ymax = max(output_ttp_all)
ymin = min(output_ttp_all)
mse = metrics.mean_squared_error(output_ttp_all, ttp_pred_all.round())
rmse = np.sqrt(mse)
nrmse_ttp = rmse/(ymax - ymin)
print('\nTTP: ', ttp_acc, nrmse_ttp)


with open(os.path.join(results_path, 'pred_results.txt'), 'w') as fp:
    fp.write('SVM mTT no paramerters specified\n')
    fp.write('\t CBF \t CBV \t MTT \t TTP\n')
    fp.write('acc: {:.2f}\t {:.2f}\t {:.2f}\t {:.2f}\n'.format(rbf_acc, rbv_acc, mtt_acc, ttp_acc))
    fp.write('nrmse: {:.2f}\t {:.2f}\t {:.2f}\t {:.2f}\n'.format(nrmse_rbf, nrmse_rbv, nrmse_mtt, nrmse_ttp))
    fp.close()

print(mtt_pred_all.shape)
# print(sum(i) if i!=0 for i in rbv_pred_all) 



rbv_map = np.reshape(rbv_pred_all, (height, width, length))
rbf_map = np.reshape(rbf_pred_all, (height, width, length))
mtt_map = np.reshape(mtt_pred_all, (height, width, length))
ttp_map = np.reshape(ttp_pred_all, (height, width, length))

print(mtt_map.shape)
print(rbv_map.shape)
# print(ttp_map.shape)


rbv_map = sitk.GetImageFromArray(rbv_map)
rbf_map = sitk.GetImageFromArray(rbf_map)
mtt_map = sitk.GetImageFromArray(mtt_map)
ttp_map = sitk.GetImageFromArray(ttp_map)

sitk.WriteImage(rbv_map, os.path.join(results_path,'maps/rbv_map.nii'))
sitk.WriteImage(rbf_map, os.path.join(results_path,'maps/rbf_map.nii'))
sitk.WriteImage(mtt_map, os.path.join(results_path,'maps/mtt_map.nii'))
sitk.WriteImage(ttp_map, os.path.join(results_path,'maps/ttp_map.nii'))

