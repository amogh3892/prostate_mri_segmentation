import SimpleITK as sitk 
from pathlib import Path
import sys 
from dataUtil import DataUtil 
import tables
import os 
import numpy as np
from glob import glob 
import matplotlib.pyplot as plt 
from augmentation3DUtil import Augmentation3DUtil
from augmentation3DUtil import Transforms

def _getAugmentedData(img,mask,nosamples):

    """ 
    This function defines different augmentations/transofrmation sepcified for a single image 
    img,mask : to be provided SimpleITK images 
    nosamples : (int) number of augmented samples to be returned
    
    """

    au = Augmentation3DUtil(img,mask=mask)

    au.add(Transforms.SHEAR,probability = 0.25, magnitude = (0.05,0.05))
    au.add(Transforms.SHEAR,probability = 0.25, magnitude = (0.02,0.05))
    au.add(Transforms.SHEAR,probability = 0.25, magnitude = (0.01,0.05))
    au.add(Transforms.SHEAR,probability = 0.25, magnitude = (0.03,0.05))

    au.add(Transforms.TRANSLATE,probability = 0.25, offset = (2,2,0))
    au.add(Transforms.TRANSLATE,probability = 0.25, offset = (-2,-2,0))
    au.add(Transforms.TRANSLATE,probability = 0.25, offset = (5,5,0))
    au.add(Transforms.TRANSLATE,probability = 0.25, offset = (-5,-5,0))

    au.add(Transforms.ROTATE2D,probability = 0.2, degrees = 3)
    au.add(Transforms.ROTATE2D,probability = 0.2, degrees = -3)
    au.add(Transforms.ROTATE2D,probability = 0.2, degrees = 5)
    au.add(Transforms.ROTATE2D,probability = 0.2, degrees = -5)
    au.add(Transforms.ROTATE2D,probability = 0.2, degrees = 8)
    au.add(Transforms.ROTATE2D,probability = 0.2, degrees = -8)

    au.add(Transforms.FLIPHORIZONTAL,probability = 0.75)

    img, augs = au.process(nosamples)

    return img,augs


def getBoundingBox(center,cs,size):

    """  
    Obtaining the bounding box for center cropping 
    center : center of the image (voxel index), 
    cs : crop size in voxels 
    size : size of the image in voxels
    """

    startz = int(center[0]-cs[2]); endz = int(center[0]+cs[2]);
    starty = int(center[1]-cs[1]); endy = int(center[1]+cs[1]);
    startx = int(center[2]-cs[0]); endx = int(center[2]+cs[0]);

    # limit the crop size within the image size 
    if startx < 0:
        startx = 0 
    if starty < 0 :
        starty = 0 
    if startz < 0:
        startz = 0 

    if endx > size[0]-1:
        endx = size[0]-1
    if endy > size[1]-1:
        endy = size[1]-1
    if endz > size[2]-1:
        endz = size[2]-1

    return startx,starty,startz,endx,endy,endz


def centerCrop(img,resample_spacing = (1,1,3),resample_interpoaltion = sitk.sitkLinear, crop_size = (60,60,60)):
    """
    img : image to be preprocessed
    resample_spacing : the resolution to be resampled to. 
    resample_interpolation : the interpolation criteria to be used for resampling. sitk.sitkLinear or sitk.sitkNearestNeighbor 
    crop_size : x,y,z crop size in mm 
    """
    origin = img.GetOrigin()
    direction = img.GetDirection()
    spacing = img.GetSpacing()
    

    cs = [(crop_size[i]/resample_spacing[i]/2) for i in range(len(crop_size))]

    # resampling to uniform resolution 
    resampled = DataUtil.resampleimage(img,resample_spacing,origin,interpolator=resample_interpoaltion)
    size = resampled.GetSize()

    # cropping image 
    imgarr = sitk.GetArrayFromImage(resampled)
    center = [int(x/2) for x in imgarr.shape]

    startx,starty,startz,endx,endy,endz = getBoundingBox(center,cs,size)
    croppedarr = imgarr[startz:endz,starty:endy,startx:endx]


    # clipping values above 99 percentile. to remove noise.
    percentileclip = np.percentile(croppedarr.ravel(),99.5)
    croppedarr = np.clip(croppedarr,0,percentileclip)

    cs_verify = tuple([int((crop_size[i]/resample_spacing[i])) for i in range(len(crop_size))])

    # if there are fewer number of slices, zeropad the image 
    if croppedarr.T.shape != cs_verify:
        diff =  cs_verify[2] - croppedarr.T.shape[2]
        zeroarr = np.zeros((diff,croppedarr.shape[1],croppedarr.shape[2]))
        croppedarr = np.vstack((croppedarr,zeroarr))

    cropped = sitk.GetImageFromArray(croppedarr)
    cropped.SetOrigin(origin)
    cropped.SetSpacing(resample_spacing)
    cropped.SetDirection(direction)

    return cropped 

def createHDF5(splitspathname,patchSize,depth):

    """
    splitspathname : name of the file (json) which has train test splits info 
    patchSize : x,y dimension of the image 
    depth : z dimension of the image 
    """
    
    outputfolder = fr"outputs\hdf5\{splitspathname}"
    Path(outputfolder).mkdir(parents=True, exist_ok=True)

    img_dtype = tables.UInt8Atom()
    pm_dtype = tables.UInt8Atom()
    data_shape = (0, depth, patchSize, patchSize)
    filters = tables.Filters(complevel=5)

    splitspath = fr"outputs\splits\{splitspathname}.json"
    splitsdict = DataUtil.readJson(splitspath)

    phases = np.unique(list(splitsdict.values()))

    for phase in phases:
        hdf5_path = fr'{outputfolder}\{phase}.h5'

        if os.path.exists(hdf5_path):
            Path(hdf5_path).unlink()

        hdf5_file = tables.open_file(hdf5_path, mode='w')


        data = hdf5_file.create_earray(hdf5_file.root, "data", img_dtype,
                                            shape=data_shape,
                                            chunkshape = (1,depth,patchSize,patchSize),
                                            filters = filters)

        mask =  hdf5_file.create_earray(hdf5_file.root, "mask", pm_dtype,
                                            shape=data_shape,
                                            chunkshape = (1,depth,patchSize,patchSize),
                                            filters = filters)

        hdf5_file.close()


def addToHDF5(img,pm,phase,splitspathname):
    
    """
    sb : Image subfolder ... path to image file 
    phase : phase of that image (train,test,val)

    """
    outputfolder = fr"outputs\hdf5\{splitspathname}"

    hdf5_file = tables.open_file(fr'{outputfolder}\{phase}.h5', mode='a')

    # img = sitk.ReadImage(fr"{str(sb)}\T2W.nii.gz")
    # pm = sitk.ReadImage(fr"{str(sb)}\PM.nii.gz")
    # pm = DataUtil.convert2binary(pm)

    imgarr = sitk.GetArrayFromImage(img)
    pmarr = sitk.GetArrayFromImage(pm)

    data = hdf5_file.root["data"]
    mask = hdf5_file.root["mask"]

    data.append(imgarr[None])
    mask.append(pmarr[None])

    hdf5_file.close()

def getAugmentedData(folderpath,modality, nosamples = None):
    
    """
    folderpath : path to folder containing images, mask 
    modality : T2W, ADC
    resample_spcing : ex (1,1,3), the resolution the images have to be resampled to. 
    crop_size : ex (60,60,60) size in mm the image to be cropped from center    
    intensitystandardize : perform intensity standardization tuple to be provided (tempimg, numberOfHistogramLevels, numberOfMatchPoints)
    """

    ext = glob(fr"{folderpath}\**")[0].split("\\")[-1].split(".")[-1]

    if ext == "gz":
        ext = ".".join(glob(fr"{folderpath}\**")[0].split("\\")[-1].split(".")[-2:])

    img = sitk.ReadImage(str(folderpath.joinpath(fr"{modality}.{ext}")))
    pm = sitk.ReadImage(str(folderpath.joinpath(fr"PM.{ext}")))
    pm = DataUtil.convert2binary(pm)

    ret = []
    
    orgimg,augs = _getAugmentedData(img,pm,nosamples)
    ret.append((orgimg))
    for i in range(len(augs)):
        ret.append(augs[i])

    return ret

def preprocessData(img,pm,resample_spacing, crop_size, intensitystandardize = None):

    """ 
    Pre-processing the images : center cropping and intensity standardization.

    img : SimpleITK image to be pre-processed
    pm : prostate mask as SimpleITK image to be preprocesed
    resample_spacing : spacing to which the image have to be resampled 
    crop_size : x,y,z crop size in mm 
    intensitystandardize : as a tuple (template image as SimpleITK image, Number of histogram levels, Number of match points)
    """

    img = centerCrop(img,resample_spacing = resample_spacing,resample_interpoaltion = sitk.sitkLinear, crop_size = crop_size)
    pm = centerCrop(pm,resample_spacing = resample_spacing,resample_interpoaltion = sitk.sitkNearestNeighbor, crop_size = crop_size)

    if intensitystandardize is not None:
        tempimg = intensitystandardize[0]
        numberOfHistogramLevels = intensitystandardize[1]
        numberOfMatchPoints = intensitystandardize[2]

        img = sitk.HistogramMatching(img,tempimg,numberOfHistogramLevels=numberOfHistogramLevels,
         numberOfMatchPoints=numberOfMatchPoints,thresholdAtMeanIntensity=False)


    img = sitk.RescaleIntensity(img)
    img = sitk.Cast(img,sitk.sitkUInt8)

    return img,pm

def _plot(img,pm,slno):
    fig = plt.figure()
    plt.subplot(121)
    plt.imshow(sitk.GetArrayFromImage(img)[slno],cmap='gray')
    plt.subplot(122)
    plt.imshow(sitk.GetArrayFromImage(pm)[slno],cmap='gray')
    plt.show()
    plt.close()

if __name__ == "__main__":

    datadir = fr"..\Data"
    modality = 'T2W'
    datasets = ["UHRD"]
    
    # for resampling 
    resample_spacing = (1,1,1)
    crop_size = (96,96,64) 
    patchSize = int(crop_size[0]/resample_spacing[0])
    depth = int(crop_size[2]/resample_spacing[2])

    # for augmentation
    nosamples = 30

    # for intensity standardization
    numberOfHistogramLevels = 1024
    numberOfMatchPoints = 5 

    # splits to create HDF5 files 
    splitspathname = fr"UH"
    splitspath = fr"outputs\splits\{splitspathname}.json"

    splitsdict = DataUtil.readJson(splitspath)
    createHDF5(splitspathname,patchSize,depth)

    casenames = {} 
    casenames["train"] = [] 
    casenames["val"] = [] 
    casenames["test"] = [] 

    for j,dataset in enumerate(datasets):

        inputfolder = fr"{datadir}\{dataset}\{modality}\1_Original_Organized"
        subpaths = DataUtil.getSubDirectories(inputfolder)

        subpaths = [x for x in subpaths if (("L2" not in str(x)) and ("L3" not in str(x)) and ("L4" not in str(x)))  ]

        for i,sb in enumerate(subpaths):
            name = sb.stem.replace("_L1","")
            print(name,float(i)/len(subpaths))


            phase = splitsdict[name]

            if (i == 0) and (j == 0) :

                if not os.path.exists(fr"Template\{modality}"):
                    os.makedirs(fr"Template\{modality}")

                ret = getAugmentedData(sb,modality,nosamples = None)

                templateimg = ret[0][0]
                templatepm = ret[0][1]

                _max = sitk.GetArrayFromImage(templateimg).max()
                _min = sitk.GetArrayFromImage(templateimg).min()

                intensitystandardize = (templateimg, numberOfHistogramLevels, numberOfMatchPoints)

                sitk.WriteImage(templateimg,fr"Template\{modality}\T2W.nii.gz")
                sitk.WriteImage(templatepm,fr"Template\{modality}\PM.nii.gz")                
            
            if phase == "train":
                ret = getAugmentedData(sb,modality,nosamples=nosamples)
            else:
                ret = getAugmentedData(sb,modality,nosamples=None)

            for k,aug in enumerate(ret):
                augimg = aug[0]
                augpm = aug[1]

                casename = name if k == 0 else fr"{name}_A{k}"
                casenames[phase].append(casename)

                pimg,ppm = preprocessData(augimg,augpm,resample_spacing, crop_size, intensitystandardize = intensitystandardize)
                
                addToHDF5(pimg,ppm,phase,splitspathname)

                
    outputfolder = fr"outputs\hdf5\{splitspathname}"

    for phase in ["train","test","val"]:
        hdf5_file = tables.open_file(fr'{outputfolder}\{phase}.h5', mode='a')
        hdf5_file.create_array(hdf5_file.root, fr'names', casenames[phase])
        hdf5_file.close()

