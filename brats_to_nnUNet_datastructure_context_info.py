'''
  Copyright 2021 Department of Biomedical Engineering, Linkoping University, Linkoping, Sweden

  Licensed under the Apache License, Version 2.0 (the "License");
  you may not use this file except in compliance with the License.
  You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

  Unless required by applicable law or agreed to in writing, software
  distributed under the License is distributed on an "AS IS" BASIS,
  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  See the License for the specific language governing permissions and
  limitations under the License.

  DESCRIPTION

  Python script for the preparation of the data following the nnU-Net dataset
  specification for the project

  'Does anatomical contextual information improve 3D U-Net based brain tumor segmentation?'

  Specification of how the data needs to be ogranized is avvailable at:
  https://github.com/MIC-DKFZ/nnUNet#dataset-conversion

  STEPS
  1 - get path to the BraTS20 subject files and the specification of which are
      LGG and HGG (this information is in the grade_split.txt file obtained from
      the available name_mapping.csv file provided with the BraTS20 dataset)

  2 - create independend test samples by randomly selecting samples from the
      cases available. Note that havign the information about which subjects
      are LGGs and which are HGGs, we can specify the percentage of LGG and HGG
      in the test set

  3 - prepare the testing dataset: for all the samples do
      - load BraTS nifty modalities
      - load contextual information if required (obtained using the
        run_contextual_segmetnation.sh script)
      - load labels
      - save data as specified by nnU-Net dataset specifications

  4 - prepare the training dataset: for all the samples do
      - load BraTS nifty modalities
      - load contextual information if required (obtained using the
        run_contextual_segmetnation.sh script)
      - load labels
      - save data as specified by nnU-Net dataset specifications

  5 - save dataset information in a .json file as specified by nnU-Net
'''


import glob
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import sys
import numpy as np
import nibabel as nib
import random
import json
from tensorflow.keras.utils import to_categorical
from collections import OrderedDict

## RETRIEVE FILE PATHS, SHUFFLE THEM and SELECT SUBJECTS FOR TEST SET

BASEFOLDER = 'where/the/BraTS20/dataset/is/located'
BASEFOLDER_CONTEXT_INFO = 'where/the/contextual/information/is/located'
BASEFOLDER_PROCESSED = 'where/to/save/the/nnUNet-like/dataset'

TASK_ID = 'TEST_001'

# create directories for saving data
os.mkdir(os.path.join(BASEFOLDER_PROCESSED, TASK_ID))
os.mkdir(os.path.join(BASEFOLDER_PROCESSED, TASK_ID, 'imagesTs'))
os.mkdir(os.path.join(BASEFOLDER_PROCESSED, TASK_ID, 'labelsTs'))
os.mkdir(os.path.join(BASEFOLDER_PROCESSED, TASK_ID, 'imagesTr'))
os.mkdir(os.path.join(BASEFOLDER_PROCESSED, TASK_ID, 'labelsTr'))


# specify BraTS modalities to use and create identification code
MODALITIES = ['t1', 't1ce', 't2', 'flair']
MODALITIES_CODE = []
for i in range(len(MODALITIES)):
  MODALITIES_CODE.append('%04d'%(i))

# specify if contextual information is needed and if yes, what type:
# - binary_mask
# - prob_maps

contextual_information = True
type_context = 'binary_mask'
# type_context = 'prob_maps'

# get information about subject grades
with open('/path/to/the/file/containing/info/about/LGGandHGG/samples/grade_split.txt') as json_file:
    grade_split = json.load(json_file)
    HGG_file = grade_split['HGG']
    LGG_file = grade_split['LGG']

subject_grade = ['HGG']*len(HGG_file) + ['LGG']*len(LGG_file)
subjectID = HGG_file + LGG_file
temp = zip(subjectID, subject_grade)
temp = sorted(temp)
subjectID, subject_grade = zip(*temp)

data_filenames = sorted(glob.glob(os.path.join(BASEFOLDER,'*')))

# split train/validation and testing filenames
# here set the test size and the % of LGG in the test dataset
# e.g. 10% for testing -> 30 samples and 50% LGG -> 15 LGG samples in the test dataset

p_testing = 10
p_LGG = 50
n_test_samples = np.floor(len(subjectID)*p_testing/100)
n_test_LGG_samples = np.floor(n_test_samples*50/100)
n_test_HGG_samples = n_test_samples - n_test_LGG_samples

# print('Total samples %d \n Total test samples %d \n LGG test samples %d \n HGG test samples %d'%(len(subjectID), n_test_samples, n_test_LGG_samples, n_test_HGG_samples))

# select randomly from the available samples
# for reproducibility, fix the random seed
np.random.seed(30)

LGG_index = [i for i, j in enumerate(subject_grade) if j == 'LGG']
HGG_index = [i for i, j in enumerate(subject_grade) if j == 'HGG']
index_LGG_test, index_LGG_train = np.split(np.random.permutation(LGG_index), [int(n_test_LGG_samples)])
index_HGG_test, index_HGG_train = np.split(np.random.permutation(HGG_index), [int(n_test_HGG_samples)])


test_filenames = [data_filenames[i] for i in index_LGG_test.tolist()+index_HGG_test.tolist()]
test_subjectID = [subjectID[i] for i in index_LGG_test.tolist()+index_HGG_test.tolist()]
test_subject_grade = [subject_grade[i] for i in index_LGG_test.tolist()+index_HGG_test.tolist()]

train_filenames = [data_filenames[i] for i in index_LGG_train.tolist()+index_HGG_train.tolist()]
train_subjectID = [subjectID[i] for i in index_LGG_train.tolist()+index_HGG_train.tolist()]
train_subject_grade = [subject_grade[i] for i in index_LGG_train.tolist()+index_HGG_train.tolist()]

# n = 35
# print('Test File name %s \n Subject ID %s \n Subject Grade %s \n'%(test_filenames[n], test_subjectID[n], test_subject_grade[n]))
# print('Train File name %s \n Subject ID %s \n Subject Grade %s \n'%(train_filenames[n], train_subjectID[n], train_subject_grade[n]))

## TESTING DATA -  open the different nifi files and save with the right naming (SubjectID_modalityCode.nii.gz)
print('Working on the testing data \n')

for s in range(len(test_subjectID)):
# for s in range(3):
  print('   Subject {}/{}'.format(s, len(test_subjectID)))
  # get info for later saving
  raw_nifti_file = nib.load(os.path.join(test_filenames[s], test_subjectID[s] + '_'+ MODALITIES[0] +'.nii'))

  # load BRATS subject data
  image_temp = []
  for mod in range(len(MODALITIES)):
    image_temp.append(nib.load(os.path.join(test_filenames[s], test_subjectID[s] + '_'+ MODALITIES[mod] +'.nii')).get_fdata(dtype='float32'))

  # add the contextual information if needed
  if contextual_information == True:
    if type_context == 'binary_mask':
      contextual_info = nib.load(os.path.join(BASEFOLDER_CONTEXT_INFO, test_subjectID[s] + '_seg.nii.gz')).get_fdata(dtype='float32')
      contextual_info = to_categorical(contextual_info, 4)
      for ci in range(1,4):
        image_temp.append(contextual_info[:,:,:,ci])

    elif type_context == 'prob_maps':
      WM_prob_map  = nib.load(os.path.join(BASEFOLDER_CONTEXT_INFO, test_subjectID[s] + '_pve_2.nii.gz')).get_fdata(dtype='float32')
      GM_prob_map  = nib.load(os.path.join(BASEFOLDER_CONTEXT_INFO, test_subjectID[s] + '_pve_1.nii.gz')).get_fdata(dtype='float32')
      CSF_prob_map = nib.load(os.path.join(BASEFOLDER_CONTEXT_INFO, test_subjectID[s] + '_pve_0.nii.gz')).get_fdata(dtype='float32')

      image_temp.extend((CSF_prob_map, GM_prob_map, WM_prob_map))
    else:
      print('Invalid contextual information type: given {}, required binary_mask or prob_maps')
      break

  # load label
  label = np.rint(nib.load(os.path.join(test_filenames[s], test_subjectID[s] + '_seg.nii')).get_fdata(dtype='float32'))
  label = label.astype(int)

  # save images with the right name as specified by nnU-Net dataset specifications
  for i in range(len(image_temp)):
    file_name = test_subjectID[s].replace('_Training','') + '_%04d.nii.gz'%(i)
    aus = nib.Nifti1Image(image_temp[i],
                          raw_nifti_file.affine,
                          raw_nifti_file.header)
    nib.save(aus, os.path.join(BASEFOLDER_PROCESSED, TASK_ID ,'imagesTs', file_name))

  # save label with the right name
  file_name = test_subjectID[s].replace('_Training','') + '.nii.gz'
  aus = nib.Nifti1Image(label,
                        raw_nifti_file.affine,
                        raw_nifti_file.header)
  nib.save(aus, os.path.join(BASEFOLDER_PROCESSED, TASK_ID, 'labelsTs', file_name))

## TRAINING DATA -  open the different nifi files and save with the right naming SubjectID_modalityCode.nii.gz
print('Working on the training data \n')

for s in range(len(train_subjectID)):
# for s in range(3):
  print('   Subject {}/{}'.format(s, len(train_subjectID)))
  # get info for later saving

  # get info for later saving
  raw_nifti_file = nib.load(os.path.join(train_filenames[s], train_subjectID[s] + '_'+ MODALITIES[0] +'.nii'))

  # load BRATS subject data
  image_temp = []
  for mod in range(len(MODALITIES)):
    image_temp.append(nib.load(os.path.join(train_filenames[s], train_subjectID[s] + '_'+ MODALITIES[mod] +'.nii')).get_fdata(dtype='float32'))

  # add the contextual information if needed
  if contextual_information == True:
    if type_context == 'binary_mask':
      contextual_info = nib.load(os.path.join(BASEFOLDER_CONTEXT_INFO, train_subjectID[s] + '_seg.nii.gz')).get_fdata(dtype='float32')
      contextual_info = to_categorical(contextual_info, 4)
      for ci in range(1,4):
        image_temp.append(contextual_info[:,:,:,ci])

    elif type_context == 'prob_maps':
      WM_prob_map  = nib.load(os.path.join(BASEFOLDER_CONTEXT_INFO, train_subjectID[s] + '_pve_2.nii.gz')).get_fdata(dtype='float32')
      GM_prob_map  = nib.load(os.path.join(BASEFOLDER_CONTEXT_INFO, train_subjectID[s] + '_pve_1.nii.gz')).get_fdata(dtype='float32')
      CSF_prob_map = nib.load(os.path.join(BASEFOLDER_CONTEXT_INFO, train_subjectID[s] + '_pve_0.nii.gz')).get_fdata(dtype='float32')

      image_temp.extend((CSF_prob_map, GM_prob_map, WM_prob_map))
    else:
      print('Invalid contextual information type: given {}, required binary_mask or prob_maps')
      break

  # load label
  label = np.rint(nib.load(os.path.join(train_filenames[s], train_subjectID[s] + '_seg.nii')).get_fdata())
  label = label.astype(int)

  # save images with the right name
  for i in range(len(image_temp)):
    file_name = train_subjectID[s].replace('_Training','') + '_%04d.nii.gz'%(i)
    aus = nib.Nifti1Image(image_temp[i],
                          raw_nifti_file.affine,
                          raw_nifti_file.header)
    nib.save(aus, os.path.join(BASEFOLDER_PROCESSED, TASK_ID, 'imagesTr', file_name))

  # save label with the right name
  file_name = train_subjectID[s].replace('_Training','') + '.nii.gz'
  aus = nib.Nifti1Image(label,
                        raw_nifti_file.affine,
                        raw_nifti_file.header)
  nib.save(aus, os.path.join(BASEFOLDER_PROCESSED, TASK_ID, 'labelsTr', file_name))


## create datase.json

aus_test_subjectID = [test_subjectID[s].replace('_Training','') for s in range(len(test_subjectID))]
aus_train_subjectID = [train_subjectID[s].replace('_Training','') for s in range(len(train_subjectID))]

json_dict = OrderedDict()
json_dict['name'] = "BraTS2020"
json_dict['description'] = "Dataset with contextual information as FSL probability maps for WM, GM and CSF"
json_dict['tensorImageSize'] = "4D"
json_dict['reference'] = "see BraTS2020"
json_dict['licence'] = "see BraTS2020 license"
json_dict['release'] = "0.0"
if contextual_information == True:
  json_dict['modality'] = {
      "0": "T1",
      "1": "T1ce",
      "2": "T2",
      "3": "FLAIR",
      "4": "CSF",
      "5": "GM",
      "6": "WM"
  }
else:
  json_dict['modality'] = {
      "0": "T1",
      "1": "T1ce",
      "2": "T2",
      "3": "FLAIR"
  }
json_dict['labels'] = {
    "0": "background",
    "1": "edema",
    "2": "non-enhancing",
    "3": "enhancing"
}
json_dict['numTraining'] = len(train_subjectID)
json_dict['numTest'] = len(test_subjectID)
json_dict['training'] = [{'image': "./imagesTr/%s.nii.gz" % i, "label": "./labelsTr/%s.nii.gz" % i}
                          for i in aus_train_subjectID]
json_dict['test'] = ["./imagesTs/%s.nii.gz" % i for i in aus_test_subjectID]

with open(os.path.join(BASEFOLDER_PROCESSED, TASK_ID,'dataset.json'), 'w') as fp:
    json.dump(json_dict, fp)

## open json ad check that everything is ok
with open(os.path.join(BASEFOLDER_PROCESSED, TASK_ID,'dataset.json'), 'r') as fp:
    data = json.load(fp)
    training = data['training']






