cd ../data/Hollywood2/AVIClips
execute_dir="../../../act_rec_code/improved_trajectory_release/release"
bb_dir="../../../data/Hollywood2_face/"
out_dir="../../../working/Hollywood2_feature_tam/"
for file in actionclipt*
do
	name=$(basename $file .avi)
	BB=${bb_dir}${name}.bb
	out_file=${out_dir}${name}.features.binary
	echo $name
#	echo $BB
	echo $out_file
	if [ ! -f $out_file ]
	then
		./$execute_dir/DenseTrackStab $file -O $out_file -H $BB
	fi
done
