close all
clear all
label_dir='../data/Hollywood2/ClipSets/';
out_file='../data/Hollywood2/labels/';
d_test=dir([label_dir,'*_test.txt']);

class_num=12;

test_label=zeros(class_num,884);
train_label=zeros(class_num,823);

for i=1:class_num
	cur_file=[label_dir,d_test(i).name];
	fid =fopen(cur_file);
	tline=fgetl(fid);
	file_index=0;;
	while ischar(tline)
		line=textscan(tline,'%s %f');
		class=line{2};
		file_index=file_index+1;
		test_label(i,file_index)=class;
		tline=fgetl(fid);
	end
	if (file_index~=884) disp('Test number not correct!');end
	fclose(fid);
end
save([out_file,'test_label.mat'],'test_label');

d_train=dir([label_dir,'*_train.txt']);

for i=1:class_num
	cur_file=[label_dir,d_train(i).name];
	fid =fopen(cur_file);
	tline=fgetl(fid);
	file_index=0;;
	while ischar(tline)
		line=textscan(tline,'%s %f');
		class=line{2};
		file_index=file_index+1;
		train_label(i,file_index)=class;
		tline=fgetl(fid);
	end
	if (file_index~=823) disp('Train number not correct!');end
	fclose(fid);
end

save([out_file,'train_label.mat'],'train_label');
