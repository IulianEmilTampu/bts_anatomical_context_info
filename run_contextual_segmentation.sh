#!/bin/bash

# BASH SCRIPT TO RUN FSL FAST algoritm on the BraTS20 dataset
# Using the input line, set the path where the BraTS20 dataset is and where to 
# output the masks



# Steps:
# 1 - take the all the subject file names
# 2 - check input paths
# 2 - run FLS FAST algorithm

# build help function

Help()
{
   # Display Help
   echo "Bash script to run FSL FAST segmentation algorithm on the BraTS20 dataset"
   echo
   echo "Syntax: run_training [d|o]"
   echo "required inputs:"
   echo "o     Path where the results of the segmetnation should be saved."
   echo "d     Path where the BraTS20 dataset is located (this is the folder where all the subjects folders are located)."
   echo
}

# ¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤ 1 - read line inputs

while getopts d:ho: option; do
case "${option}" in
   h) # display Help
       Help
       exit;;
   d) DATASET_DIR=${OPTARG};;
   o) SAVE_DIR=${OPTARG};;

   \?) # incorrect option
         echo "Error: Invalid input"
         exit 1
esac
done

# ¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤ 2 - check input paths
if ! [ -d $DATASET_DIR ]; then
     printf " $DATASET_DIR \n"
     echo " Dataset folder does not exist. Input a valid directory"
     exit 1
fi

if ! [ -d $SAVE_DIR ]; then
    printf " $SAVE_DIR \n"
    echo 'Save folder doe not exists. Input a valid directory'
    exit 1
fi


# ¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤ 3 - run FSL FAST segmentation
for i in $DATASET_DIR/*; do
   # folder to process
   SubjectID=${i##*/}  # get subject folder name
   printf "Processing folder $SubjectID \n"
   # run segmentation
   /usr/local/fsl/bin/fast -S 1 -n 3 -H 0.3 -I 4 -l 20.0 -o ${i}/${SubjectID}_t1.nii ${SAVE_DIR}/${SubjectID}
done





