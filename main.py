'''

Machine Learning Models trained on ISLES2018 CTP data

SVM(linear and rbf)

2019-12-11 by Yupei

'''

# import pydicom 
# from datetime import datetime 
import numpy as np 
import random 
from sklearn import metrics
from sklearn.model_selection import GroupKFold 
import matplotlib.pyplot as plt 
from lupy import *
import SimpleITK as sitk 
from sklearn import svm
from joblib import dump
import argparse
import yaml

# load params in config
parser = argparse.ArgumentParser(description='Temporal ensambling')
parser.add_argument('--config', default='config.yaml')
args = parser.parse_args()
with open(args.config) as f:
    config = yaml.load(f)
# for key in config:
for k, v in config.items():
    setattr(args, k, v) 


complied_arr = []# Final array to hold alll imput&output data from all patients
arr_input = []
arr_ttp = []
arr_rbf = []
arr_rbv = []
arr_mtt = []
arr_tmax = []
arr_length = [] 


numPoints = args.numPoints
numTestFiles = args.numTestFiles
data_path = args.data_path_local
results_path = args.results_path

files = fsearch(path = data_path, suffix = '.nii', include='4DPWI', sort_level=-3)
cbv_path = fsearch(path = data_path, suffix = '.nii', include='CBV', sort_level=-3)
cbf_path = fsearch(path = data_path, suffix = '.nii', include='CBF', sort_level=-3)
mtt_path = fsearch(path = data_path, suffix = '.nii', include='MTT', sort_level=-3)
tmax_path = fsearch(path = data_path, suffix = '.nii', include='Tmax', sort_level=-3)


numFiles = len(files) 
j = 0  # count the number of processing files

for file in files:
    # print(file.split('/')[-3])
    i = 0
    val_arr = []
    point_dict = {}

    img = sitk.ReadImage(file)
    img_arr = sitk.GetArrayFromImage(img) #(49, 8, 256, 256)
    if len(img_arr.shape) < 4:
        print('Error: Invalid input data shape!')
        continue

    [t, height, width, length] = img_arr.shape

    img_cbv = sitk.ReadImage(cbv_path[j])
    img_cbf = sitk.ReadImage(cbf_path[j])
    img_mtt = sitk.ReadImage(mtt_path[j])
    img_tmax = sitk.ReadImage(tmax_path[j])

    img_cbv_arr = sitk.GetArrayFromImage(img_cbv)
    img_cbf_arr = sitk.GetArrayFromImage(img_cbf) 
    img_mtt_arr = sitk.GetArrayFromImage(img_mtt) 
    img_tmax_arr = sitk.GetArrayFromImage(img_tmax) 

    img_mtt_arr = np.round(img_mtt_arr)
    img_tmax_arr = np.round(img_tmax_arr)

    img_mtt_arr = np.ndarray.astype(img_mtt_arr, 'int')
    img_tmax_arr= np.ndarray.astype(img_tmax_arr, 'int')

    print(img_mtt_arr.max(), img_mtt_arr.min())
    print(img_tmax_arr.max(), img_tmax_arr.min())

    while(i < numPoints): 
        if (j+1)%20 == 0 and (i+1)%50 ==0: 
            print('reading file {} point {}'.format(j+1, i+1))  

        #find a random x, y, z value
        x = random.randint(60, width-60)
        y = random.randint(60, length-60)
        z = random.randint(0, height-1)

        # if this combindation has already been used, make a new random combination
        while (x,y,z) in point_dict:
            x = random.randint(60, width-60)
            y = random.randint(60, length-60)
            z = random.randint(0, height-1)

        point_dict[(x,y,z)] = "True"
        x_loc = x
        y_loc = y
        sliceNum = z        

        ######################################################################## 

        # get perfusion data
        xp = []
        fp = []
        for time in range(t):
            xp.append(time)
            fp.append(img_arr[time, sliceNum, x_loc, y_loc])

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
        i += 1
    j += 1 
    if j+1 > numTestFiles:
        break


min_length = min(arr_length)

index = arr_length.index(min_length)

tempi = [t[:min_length] for t in arr_input]

arr_input = tempi

print(len(arr_input)) #2650 # 50 points * 51 files
factor = int(len(arr_input)*0.8)

#ALL DATA ARRAYS for 50 points
input_train = arr_input[0:factor]
input_test = arr_input[factor:len(arr_input)]
output_ttp_train = arr_tmax[0:factor]
output_ttp_test = arr_tmax[factor:len(arr_input)]
output_rbv_train = arr_rbv[0:factor]
output_rbv_test = arr_rbv[factor:len(arr_input)]
output_rbf_train = arr_rbf[0:factor]
output_rbf_test = arr_rbf[factor:len(arr_input)]
output_mtt_train = arr_mtt[0:factor]
output_mtt_test = arr_mtt[factor:len(arr_input)] 

# print(output_ttp_train)
# print(output_mtt_train)

groups = []
for groupNumber in range(numFiles):
    for p in range(numPoints):
        groups.append(groupNumber)
group_train = groups[0:len(input_train)]
gkf = GroupKFold(n_splits = int(len(input_train)/numPoints))


# # #SVMs
# print("////////////////////SVM linear////////////////////")

# svc_mtt = svm.SVC(kernel='linear')
# svc_mtt.fit(input_train, output_mtt_train)
# svc_mtt_pred = svc_mtt.predict(input_test)
# print("SVM LINEAR mTT")
# svc_mtt_acc = metrics.accuracy_score(output_mtt_test, svc_mtt_pred.round())
# print(svc_mtt_acc)
# ymax = max(output_mtt_test)
# ymin = min(output_mtt_test)
# mse = metrics.mean_squared_error(output_mtt_test, svc_mtt_pred.round())
# rmse = np.sqrt(mse)
# nrmse = rmse/(ymax - ymin)
# print(mse, rmse, nrmse)

# svc_ttp = svm.SVC(kernel='linear')
# svc_ttp.fit(input_train, output_ttp_train)
# svc_ttp_pred = svc_ttp.predict(input_test)
# print("SVM LINEAR TTP")
# svc_ttp_acc = metrics.accuracy_score(output_ttp_test, svc_ttp_pred.round())
# print(svc_ttp_acc)
# ymax = max(output_ttp_test)
# ymin = min(output_ttp_test)
# mse = metrics.mean_squared_error(output_ttp_test, svc_ttp_pred.round())
# rmse = np.sqrt(mse)
# nrmse = rmse/(ymax - ymin)
# print(mse, rmse, nrmse)

# svc_rbf = svm.SVC(kernel='linear')
# svc_rbf.fit(input_train, output_rbf_train)
# svc_rbf_pred = svc_rbf.predict(input_test)
# print("SVM LINEAR rBF")
# svc_rbf_acc = metrics.accuracy_score(output_rbf_test, svc_rbf_pred.round())
# print(svc_rbf_acc)
# ymax = max(output_rbf_test)
# ymin = min(output_rbf_test)
# mse = metrics.mean_squared_error(output_rbf_test, svc_rbf_pred.round())
# rmse = np.sqrt(mse)
# nrmse = rmse/(ymax - ymin)
# print(mse, rmse, nrmse)

# svc_rbv = svm.SVC(kernel='linear')
# svc_rbv.fit(input_train, output_rbv_train)
# svc_rbv_pred = svc_rbv.predict(input_test)
# print("SVM LINEAR rBV")
# svc_rbv_acc = metrics.accuracy_score(output_rbv_test, svc_rbv_pred.round())
# print(svc_rbv_acc)
# ymax = max(output_rbv_test)
# ymin = min(output_rbv_test)
# mse = metrics.mean_squared_error(output_rbv_test, svc_rbv_pred.round())
# rmse = np.sqrt(mse)
# nrmse = rmse/(ymax - ymin)
# print(mse, rmse, nrmse)


print("////////////////////SVM rbf////////////////////") 

arr_svc_ttp_acc_c = []
arr_svc_rbf_acc_c = []
arr_svc_rbv_acc_c = []
arr_svc_mtt_acc_c = []
arr_svc_ttp_acc_g = []
arr_svc_rbf_acc_g = []
arr_svc_rbv_acc_g = []
arr_svc_mtt_acc_g = []

C_range = 10.0 ** np.arange(-2, 2)
gamma_range = [0.15, 0.1, 0.05, 0.01]
print("fitting for ttp constants")
for c in C_range:
    svc_ttp = svm.SVC(C=c, kernel='rbf')
    temp_acc = []
    for train_index, test_index in gkf.split(input_train, output_ttp_train, groups=group_train):
        new_input_train = []
        new_output_train = []
        new_input_cv = []
        new_output_cv = []
        for i in train_index:
            new_input_train.append(input_train[i])
            new_output_train.append(output_ttp_train[i])
        for i in test_index:
            new_input_cv.append(input_train[i])
            new_output_cv.append(output_ttp_train[i])
        svc_ttp.fit(new_input_train, new_output_train)
        svc_ttp_pred = svc_ttp.predict(new_input_cv)
        svc_ttp_acc = metrics.accuracy_score(new_output_cv, svc_ttp_pred.round())
        temp_acc.append(svc_ttp_acc)
    arr_svc_ttp_acc_c.append(np.mean(temp_acc))
print("fitting for ttp gammas")
for g in gamma_range:
    svc_ttp = svm.SVC(kernel='rbf', gamma=g)
    temp_acc = []
    for train_index, test_index in gkf.split(input_train, output_ttp_train, groups=group_train):
        new_input_train = []
        new_output_train = []
        new_input_cv = []
        new_output_cv = []
        for i in train_index:
            new_input_train.append(input_train[i])
            new_output_train.append(output_ttp_train[i])
        for i in test_index:
            new_input_cv.append(input_train[i])
            new_output_cv.append(output_ttp_train[i])
        svc_ttp.fit(new_input_train, new_output_train)
        svc_ttp_pred = svc_ttp.predict(new_input_cv)
        svc_ttp_acc = metrics.accuracy_score(new_output_cv, svc_ttp_pred.round())
        temp_acc.append(svc_ttp_acc)
    arr_svc_ttp_acc_g.append(np.mean(temp_acc))
print("fitting for rbf constants")
for c in C_range:
    svc_rbf = svm.SVC(C=c, kernel='rbf')
    temp_acc = []
    for train_index, test_index in gkf.split(input_train, output_rbf_train, groups=group_train):
        new_input_train = []
        new_output_train = []
        new_input_cv = []
        new_output_cv = []
        for i in train_index:
            new_input_train.append(input_train[i])
            new_output_train.append(output_rbf_train[i])
        for i in test_index:
            new_input_cv.append(input_train[i])
            new_output_cv.append(output_rbf_train[i])
        svc_rbf.fit(new_input_train, new_output_train)
        svc_rbf_pred = svc_rbf.predict(new_input_cv)
        svc_rbf_acc = metrics.accuracy_score(new_output_cv, svc_rbf_pred.round())
        temp_acc.append(svc_rbf_acc)
    arr_svc_rbf_acc_c.append(np.mean(temp_acc))
print("fitting for rbf gammas")
for g in gamma_range:
    svc_rbf = svm.SVC(kernel='rbf', gamma=g)
    temp_acc = []
    for train_index, test_index in gkf.split(input_train, output_rbf_train, groups=group_train):
        new_input_train = []
        new_output_train = []
        new_input_cv = []
        new_output_cv = []
        for i in train_index:
            new_input_train.append(input_train[i])
            new_output_train.append(output_rbf_train[i])
        for i in test_index:
            new_input_cv.append(input_train[i])
            new_output_cv.append(output_rbf_train[i])
        svc_rbf.fit(new_input_train, new_output_train)
        svc_rbf_pred = svc_rbf.predict(new_input_cv)
        svc_rbf_acc = metrics.accuracy_score(new_output_cv, svc_rbf_pred.round())
        temp_acc.append(svc_rbf_acc)
    arr_svc_rbf_acc_g.append(np.mean(temp_acc))
print("fitting for rbv constants")
for c in C_range:
    svc_rbv = svm.SVC(C=c, kernel='rbf')
    temp_acc = []
    for train_index, test_index in gkf.split(input_train, output_rbv_train, groups=group_train):
        new_input_train = []
        new_output_train = []
        new_input_cv = []
        new_output_cv = []
        for i in train_index:
            new_input_train.append(input_train[i])
            new_output_train.append(output_rbv_train[i])
        for i in test_index:
            new_input_cv.append(input_train[i])
            new_output_cv.append(output_rbv_train[i])
        svc_rbv.fit(new_input_train, new_output_train)
        svc_rbv_pred = svc_rbv.predict(new_input_cv)
        svc_rbv_acc = metrics.accuracy_score(new_output_cv, svc_rbv_pred.round())
        temp_acc.append(svc_rbv_acc)
    arr_svc_rbv_acc_c.append(np.mean(temp_acc))
print("fitting for rbv gammas")
for g in gamma_range:
    svc_rbv = svm.SVC(kernel='rbf', gamma=g)
    temp_acc = []
    for train_index, test_index in gkf.split(input_train, output_rbv_train, groups=group_train):
        new_input_train = []
        new_output_train = []
        new_input_cv = []
        new_output_cv = []
        for i in train_index:
            new_input_train.append(input_train[i])
            new_output_train.append(output_rbv_train[i])
        for i in test_index:
            new_input_cv.append(input_train[i])
            new_output_cv.append(output_rbv_train[i])
        svc_rbv.fit(new_input_train, new_output_train)
        svc_rbv_pred = svc_rbv.predict(new_input_cv)
        svc_rbv_acc = metrics.accuracy_score(new_output_cv, svc_rbv_pred.round())
        temp_acc.append(svc_rbv_acc)
    arr_svc_rbv_acc_g.append(np.mean(temp_acc))
print("fitting for mtt constants")
for c in C_range:
    svc_mtt = svm.SVC(C=c, kernel='rbf')
    temp_acc = []
    for train_index, test_index in gkf.split(input_train, output_mtt_train, groups=group_train):
        new_input_train = []
        new_output_train = []
        new_input_cv = []
        new_output_cv = []
        for i in train_index:
            new_input_train.append(input_train[i])
            new_output_train.append(output_mtt_train[i])
        for i in test_index:
            new_input_cv.append(input_train[i])
            new_output_cv.append(output_mtt_train[i])
        svc_mtt.fit(new_input_train, new_output_train)
        svc_mtt_pred = svc_mtt.predict(new_input_cv)
        svc_mtt_acc = metrics.accuracy_score(new_output_cv, svc_mtt_pred.round())
        temp_acc.append(svc_mtt_acc)
    arr_svc_mtt_acc_c.append(np.mean(temp_acc))
print("fitting for mtt gammas")
for g in gamma_range:
    svc_mtt = svm.SVC(kernel='rbf', gamma=g)
    temp_acc = []
    for train_index, test_index in gkf.split(input_train, output_mtt_train, groups=group_train):
        new_input_train = []
        new_output_train = []
        new_input_cv = []
        new_output_cv = []
        for i in train_index:
            new_input_train.append(input_train[i])
            new_output_train.append(output_mtt_train[i])
        for i in test_index:
            new_input_cv.append(input_train[i])
            new_output_cv.append(output_mtt_train[i])
        svc_mtt.fit(new_input_train, new_output_train)
        svc_mtt_pred = svc_mtt.predict(new_input_cv)
        svc_mtt_acc = metrics.accuracy_score(new_output_cv, svc_mtt_pred.round())
        temp_acc.append(svc_mtt_acc)
    arr_svc_mtt_acc_g.append(np.mean(temp_acc))


xs_g = gamma_range
xs_c = C_range

plt.figure(5)
plt.plot(xs_g, arr_svc_ttp_acc_g)
plt.savefig(os.path.join(results_path,'gamma_c/svc_ttp_acc_gamma.png'))
plt.figure(6)
plt.plot(xs_c, arr_svc_ttp_acc_c)
plt.savefig(os.path.join(results_path,'gamma_c/svc_ttp_acc_c.png'))

plt.figure(7)
plt.plot(xs_g, arr_svc_rbf_acc_g)
plt.savefig(os.path.join(results_path,'gamma_c/svc_rbf_acc_gamma.png'))
plt.figure(8)
plt.plot(xs_c, arr_svc_rbf_acc_c)
plt.savefig(os.path.join(results_path, 'gamma_c/svc_rbf_acc_c.png'))

plt.figure(9)
plt.plot(xs_g, arr_svc_rbv_acc_g)
plt.savefig(os.path.join(results_path, 'gamma_c/svc_rbv_acc_gamma.png'))
plt.figure(10)
plt.plot(xs_c, arr_svc_rbv_acc_c)
plt.savefig(os.path.join(results_path, 'gamma_c/svc_rbv_acc_c.png'))

plt.figure(11)
plt.plot(xs_g, arr_svc_mtt_acc_g)
plt.savefig(os.path.join(results_path, 'gamma_c/svc_mtt_acc_gamma.png'))
plt.figure(12)
plt.plot(xs_c, arr_svc_mtt_acc_c)
plt.savefig(os.path.join(results_path, 'gamma_c/svc_mtt_acc_c.png'))

max_ttp_acc_c = max(arr_svc_ttp_acc_c)
index_max_ttp_acc_c = arr_svc_ttp_acc_c.index(max_ttp_acc_c)
max_rbf_acc_c = max(arr_svc_rbf_acc_c)
index_max_rbf_acc_c = arr_svc_rbf_acc_c.index(max_rbf_acc_c)
max_rbv_acc_c = max(arr_svc_rbv_acc_c)
index_max_rbv_acc_c = arr_svc_rbv_acc_c.index(max_rbv_acc_c)
max_mtt_acc_c = max(arr_svc_mtt_acc_c)
index_max_mtt_acc_c = arr_svc_mtt_acc_c.index(max_mtt_acc_c)

max_ttp_acc_g = max(arr_svc_ttp_acc_g)
index_max_ttp_acc_g = arr_svc_ttp_acc_g.index(max_ttp_acc_g)
max_rbf_acc_g = max(arr_svc_rbf_acc_g)
index_max_rbf_acc_g = arr_svc_rbf_acc_g.index(max_rbf_acc_g)
max_rbv_acc_g = max(arr_svc_rbv_acc_g)
index_max_rbv_acc_g = arr_svc_rbv_acc_g.index(max_rbv_acc_g)
max_mtt_acc_g = max(arr_svc_mtt_acc_g)
index_max_mtt_acc_g = arr_svc_mtt_acc_g.index(max_mtt_acc_g)

svc_ttp = svm.SVC(C=C_range[index_max_ttp_acc_c], kernel='rbf', gamma=gamma_range[index_max_ttp_acc_g])
svc_ttp.fit(input_train, output_ttp_train)
svc_ttp_pred = svc_ttp.predict(input_test)
print("SVM TTP using c={} gamma={}".format(C_range[index_max_ttp_acc_c], gamma_range[index_max_ttp_acc_g]))
svc_ttp_acc = metrics.accuracy_score(output_ttp_test, svc_ttp_pred.round()) # accuracy for? 
print(svc_ttp_acc)
ymax = max(output_ttp_test)
ymin = min(output_ttp_test)
mse = metrics.mean_squared_error(output_ttp_test, svc_ttp_pred.round())
rmse = np.sqrt(mse)
nrmse_ttp = rmse/(ymax - ymin)
print(nrmse_ttp)

svc_rbf = svm.SVC(C=C_range[index_max_rbf_acc_c], kernel='rbf', gamma=gamma_range[index_max_rbf_acc_g])
svc_rbf.fit(input_train, output_rbf_train)
svc_rbf_pred = svc_rbf.predict(input_test)
print("SVM rBF using c={}, gamma={}".format(C_range[index_max_rbf_acc_c], gamma_range[index_max_rbf_acc_g]))
svc_rbf_acc = metrics.accuracy_score(output_rbf_test, svc_rbf_pred.round())
print(svc_rbf_acc)
ymax = max(output_rbf_test)
ymin = min(output_rbf_test)
mse = metrics.mean_squared_error(output_rbf_test, svc_rbf_pred.round())
rmse = np.sqrt(mse)
nrmse_rbf = rmse/(ymax - ymin)
print(nrmse_rbf)

svc_rbv = svm.SVC(C=C_range[index_max_rbv_acc_c], kernel='rbf', gamma=gamma_range[index_max_rbv_acc_g])
svc_rbv.fit(input_train, output_rbv_train)
svc_rbv_pred = svc_rbv.predict(input_test)
print("SVM rBV using c={}, gamma={}".format(C_range[index_max_rbv_acc_c], gamma_range[index_max_rbv_acc_g]))
svc_rbv_acc = metrics.accuracy_score(output_rbv_test, svc_rbv_pred.round())
print(svc_rbv_acc)
ymax = max(output_rbv_test)
ymin = min(output_rbv_test)
mse = metrics.mean_squared_error(output_rbv_test, svc_rbv_pred.round())
rmse = np.sqrt(mse)
nrmse_rbv = rmse/(ymax - ymin)
print(nrmse_rbv)

svc_mtt = svm.SVC(C=C_range[index_max_mtt_acc_c], kernel='rbf', gamma=gamma_range[index_max_mtt_acc_g])
svc_mtt.fit(input_train, output_mtt_train)
svc_mtt_pred = svc_mtt.predict(input_test)
print("SVM mTT using c={}, gamma={}".format(C_range[index_max_mtt_acc_c], gamma_range[index_max_mtt_acc_g]))
svc_mtt_acc = metrics.accuracy_score(output_mtt_test, svc_mtt_pred.round())
print(svc_mtt_acc)
ymax = max(output_mtt_test)
ymin = min(output_mtt_test)
mse = metrics.mean_squared_error(output_mtt_test, svc_mtt_pred.round())
rmse = np.sqrt(mse)
nrmse_mtt = rmse/(ymax - ymin)
print(nrmse_mtt)

dump(svc_mtt, os.path.join(results_path, 'models/mtt.joblib'))
dump(svc_ttp, os.path.join(results_path, 'models/ttp.joblib'))
dump(svc_rbv, os.path.join(results_path, 'models/rbv.joblib'))
dump(svc_rbf, os.path.join(results_path, 'models/rbf.joblib'))

with open(os.path.join(results_path, 'metrics.txt'), 'w') as fp: 
    fp.write('SVM mTT with paramerters specified\n')
    fp.write("SVM rBF using c={}, gamma={}\n".format(C_range[index_max_rbf_acc_c], gamma_range[index_max_rbf_acc_g]))
    fp.write("SVM rBV using c={}, gamma={}\n".format(C_range[index_max_rbv_acc_c], gamma_range[index_max_rbv_acc_g]))
    fp.write("SVM mTT using c={}, gamma={}\n".format(C_range[index_max_mtt_acc_c], gamma_range[index_max_mtt_acc_g]))
    fp.write("SVM TTP using c={} gamma={}\n".format(C_range[index_max_ttp_acc_c], gamma_range[index_max_ttp_acc_g]))
    fp.write('\t CBF \t CBV \t MTT \t TTP\n')
    fp.write('acc: {:.2f}\t {:.2f}\t {:.2f}\t {:.2f}\n'.format(svc_rbf_acc, svc_rbv_acc, svc_mtt_acc, svc_ttp_acc))
    fp.write('nrmse: {:.2f}\t {:.2f}\t {:.2f}\t {:.2f}\n\n'.format(nrmse_rbf, nrmse_rbv, nrmse_mtt, nrmse_ttp))
    fp.close()

### auto

svc_ttp = svm.SVC(kernel='rbf')
svc_ttp.fit(input_train, output_ttp_train)
svc_ttp_pred = svc_ttp.predict(input_test)
print("SVM TTP no paramerters specified")
svc_ttp_acc = metrics.accuracy_score(output_ttp_test, svc_ttp_pred.round())
print(svc_ttp_acc)
ymax = max(output_ttp_test)
ymin = min(output_ttp_test)
mse = metrics.mean_squared_error(output_ttp_test, svc_ttp_pred.round())
rmse = np.sqrt(mse)
nrmse_ttp = rmse/(ymax - ymin)
print(nrmse_ttp)

svc_rbf = svm.SVC(kernel='rbf')
svc_rbf.fit(input_train, output_rbf_train)
svc_rbf_pred = svc_rbf.predict(input_test)
print("SVM rBF no paramerters specified")
svc_rbf_acc = metrics.accuracy_score(output_rbf_test, svc_rbf_pred.round())
print(svc_rbf_acc)
ymax = max(output_rbf_test)
ymin = min(output_rbf_test)
mse = metrics.mean_squared_error(output_rbf_test, svc_rbf_pred.round())
rmse = np.sqrt(mse)
nrmse_rbf = rmse/(ymax - ymin)
print(nrmse_rbf)

svc_rbv = svm.SVC(kernel='rbf')
svc_rbv.fit(input_train, output_rbv_train)
svc_rbv_pred = svc_rbv.predict(input_test)
print("SVM rBV no paramerters specified")
svc_rbv_acc = metrics.accuracy_score(output_rbv_test, svc_rbv_pred.round())
print(svc_rbv_acc)
ymax = max(output_rbv_test)
ymin = min(output_rbv_test)
mse = metrics.mean_squared_error(output_rbv_test, svc_rbv_pred.round())
rmse = np.sqrt(mse)
nrmse_rbv = rmse/(ymax - ymin)
print(nrmse_rbv)

svc_mtt = svm.SVC(kernel='rbf')
svc_mtt.fit(input_train, output_mtt_train)
svc_mtt_pred = svc_mtt.predict(input_test)
print("SVM mTT no paramerters specified")
svc_mtt_acc = metrics.accuracy_score(output_mtt_test, svc_mtt_pred.round())
print(svc_mtt_acc)
ymax = max(output_mtt_test)
ymin = min(output_mtt_test)
mse = metrics.mean_squared_error(output_mtt_test, svc_mtt_pred.round())
rmse = np.sqrt(mse)
nrmse_mtt = rmse/(ymax - ymin)
print(nrmse_mtt)


dump(svc_mtt, os.path.join(results_path, 'models/mtt_auto.joblib'))
dump(svc_ttp, os.path.join(results_path, 'models/ttp_auto.joblib'))
dump(svc_rbv, os.path.join(results_path, 'models/rbv_auto.joblib'))
dump(svc_rbf, os.path.join(results_path, 'models/rbf_auto.joblib'))


with open(os.path.join(results_path, 'metrics.txt'), 'a') as fp:
    fp.write('SVM mTT no paramerters specified\n')
    fp.write('CBF \t CBV \t MTT \t TTP\n')
    fp.write('acc: {:.2f}\t {:.2f}\t {:.2f}\t {:.2f}\n'.format(svc_rbf_acc, svc_rbv_acc, svc_mtt_acc, svc_ttp_acc))
    fp.write('nrmse: {:.2f}\t {:.2f}\t {:.2f}\t {:.2f}\n'.format(nrmse_rbf, nrmse_rbv, nrmse_mtt, nrmse_ttp))
    fp.close()
