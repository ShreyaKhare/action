action
======
This code is a full version of the algorithms in "Action Recognitionwith Improved Trajectories" Wang Heng et.al. ICCV2013

The source code from the original author is modified so that the output features are in binary format. 
To extract features, run the .sh file. Please make sure the video dataset and face bounding box are in the corresponding directory.
The Hollywood2 dataset can be downloaded from http://www.di.ens.fr/~laptev/actions/hollywood2/
The human bounding box can be downloaded from http://lear.inrialpes.fr/people/wang/download/bb_file.tar.gz
Put them in data/ after downloading

train_imp_traj.m converts the raw features to fisher vector, trains a linear SVM for Hollywood2 and calculate mAP.

This release depends on liblinear and yael_matlab, which are all included in the package.
