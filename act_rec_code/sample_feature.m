function [samples]=sample_feature(N,feature_dir,out_dir)
    d=dir([feature_dir,'actioncliptrain*']);
    file_num=length(d);
    feat_num=zeros(1,file_num);
    for i=1:file_num
        cur_file=[feature_dir,d(i).name];
        fid=fopen(cur_file);
        feat_num(i)=fread(fid,1,'float');
        fclose(fid);
    end
    
    fwid=fopen([out_dir,'sample_train_',num2str(N),'.features.binary'],'w');
    fwrite(fwid,N,'float');
    fwrite(fwid,436,'float');

    N_feat=sum(feat_num);
    samp_num=zeros(1,file_num);
%dbstop in sample_feature at 13
    samples=zeros(436,N);
    for i=1:file_num
        cur_file=[feature_dir,d(i).name];
        samp_num(i)=round((N-sum(samp_num(1:i-1)))*feat_num(i)/sum(feat_num(i:end)));
        cur_features=read_feature(cur_file);
        if (length(cur_features(:,1))~=436) disp('Dimension of feature is wrong'); end
        rand_ind=randperm(feat_num(i));

        samples(:,sum(samp_num(1:i-1))+1:sum(samp_num(1:i)))=cur_features(:,rand_ind(1:samp_num(i)));

        disp(['Finishe sampling for ',d(i).name])
    end
    if (sum(samp_num)~=N) disp('Number of total samples is wrong') ;end

    fwrite(fwid,reshape(samples,1,N*436),'float');
    fclose(fwid);

end
