'''
Preprocessing for medical data

2019-12-17

by Yupei
'''
import SimpleITK as sitk



#load image
path = '../'
img = sotk.ReadImage(path)
arr = sitk.GetArrayFromImage(img)


frames = 5

mask = np.ones(arr.shape)

# find brain mask
loth = 0
hith = 600
flag = any(arr < loth, arr > hith)
mask[flag] = 0

# spatial filtering



# segment out bones and soft tissues

low = 10 # soft tissue
high = 600 # bones

arr[arr < low] = 0
arr[arr > high] = 0


# interpolation



# contrast concentration(substract base image)

avg = np.mean(arr(0:frames-1, :, :, :), 0)

for i in range(arr.shape[0]):
	arr = arr(i,:,:,:) - avg(i,:,:,:)


# augmentation




