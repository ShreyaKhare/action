% Computer Fisher vector and train a linear SVM model

clear all
close all
path_config
% Configure

%% Sample 100000 feature points to estimate PCA
% One set of coefficients for Trajectory, HOG, HOF, MBHx, MBHy

% directory of saved feature
feature_dir='../working/Hollywood2_improved_trajectory_features/';
% directory of models
out_dir='../working/models/';
% directory of fisher vector
fisher_dir='../working/Hollywood2_fisher_vector/';

N_pca=256000;
resample_pca=(exist([out_dir,'sample_train_',num2str(N_pca),'.features.binary'])~=2);
recal_pca=0;

if resample_pca==1
	disp('Resampling for PCA')
	X_sample=single(sample_feature(N_pca,feature_dir,out_dir));
	disp('Finish sampling for PCA')
else
	X_sample=single(read_feature([out_dir,'sample_train_',num2str(N_pca),'.features.binary']));
    disp('Samples loaded')
end

if recal_pca==1
	[pca1.X, pca1.eigvec, pca1.eigval, pca1.mu] = yael_pca (X_sample(11:40,:), 15, 'center', 'verbose');
	[pca2.X, pca2.eigvec, pca2.eigval, pca2.mu] = yael_pca (X_sample(41:136,:), 48, 'center', 'verbose');
	[pca3.X, pca3.eigvec, pca3.eigval, pca3.mu] = yael_pca (X_sample(137:244,:), 54, 'center', 'verbose');
	[pca4.X, pca4.eigvec, pca4.eigval, pca4.mu] = yael_pca (X_sample(245:340,:), 48, 'center', 'verbose');
    [pca5.X, pca5.eigvec, pca5.eigval, pca5.mu] = yael_pca (X_sample(341:436,:), 48, 'center', 'verbose');
	
	save([out_dir,'pca1_',num2str(N_pca),'.mat'],'pca1');
	save([out_dir,'pca2_',num2str(N_pca),'.mat'],'pca2');
	save([out_dir,'pca3_',num2str(N_pca),'.mat'],'pca3');
	save([out_dir,'pca4_',num2str(N_pca),'.mat'],'pca4');
  	save([out_dir,'pca5_',num2str(N_pca),'.mat'],'pca5');
	disp('Finish PCA')
else
	load([out_dir,'pca1_',num2str(N_pca),'.mat']);
	load([out_dir,'pca2_',num2str(N_pca),'.mat']);
	load([out_dir,'pca3_',num2str(N_pca),'.mat']);
	load([out_dir,'pca4_',num2str(N_pca),'.mat']);
    load([out_dir,'pca5_',num2str(N_pca),'.mat']);
	disp('PCA loaded')
end

%% Estimate GMM
K=256;
if recal_pca==1
	X_sample_gmm_pca1=pca1.eigvec'*(X_sample(11:40,:)-repmat(pca1.mu,1,N_pca));
	X_sample_gmm_pca2=pca2.eigvec'*(X_sample(41:136,:)-repmat(pca2.mu,1,N_pca));
	X_sample_gmm_pca3=pca3.eigvec'*(X_sample(137:244,:)-repmat(pca3.mu,1,N_pca));
	X_sample_gmm_pca4=pca4.eigvec'*(X_sample(245:340,:)-repmat(pca4.mu,1,N_pca));
    X_sample_gmm_pca5=pca5.eigvec'*(X_sample(341:436,:)-repmat(pca5.mu,1,N_pca));

	[gmm1.w, gmm1.mu, gmm1.sigma] = yael_gmm (X_sample_gmm_pca1, K);
	[gmm2.w, gmm2.mu, gmm2.sigma] = yael_gmm (X_sample_gmm_pca2, K);
	[gmm3.w, gmm3.mu, gmm3.sigma] = yael_gmm (X_sample_gmm_pca3, K);
	[gmm4.w, gmm4.mu, gmm4.sigma] = yael_gmm (X_sample_gmm_pca4, K);
    [gmm5.w, gmm5.mu, gmm5.sigma] = yael_gmm (X_sample_gmm_pca5, K);
	clear X_sample_gmm*
	disp('Finish GMM estimation')

	save([out_dir,'gmm1_pca',num2str(N_pca),'.mat'],'gmm1'); 
	save([out_dir,'gmm2_pca',num2str(N_pca),'.mat'],'gmm2');
	save([out_dir,'gmm3_pca',num2str(N_pca),'.mat'],'gmm3');
	save([out_dir,'gmm4_pca',num2str(N_pca),'.mat'],'gmm4');
    save([out_dir,'gmm5_pca',num2str(N_pca),'.mat'],'gmm5');

else
	load ([out_dir,'gmm1_pca',num2str(N_pca),'.mat']);
	load ([out_dir,'gmm2_pca',num2str(N_pca),'.mat']);
	load ([out_dir,'gmm3_pca',num2str(N_pca),'.mat']);
	load ([out_dir,'gmm4_pca',num2str(N_pca),'.mat']);
 	load ([out_dir,'gmm5_pca',num2str(N_pca),'.mat']);
	disp('GMM loaded')
end
        
%% Calculate Fisher vector for each video

d=dir([feature_dir,'actioncliptrain*.binary']);
num_file=length(d);
fisher_dim=(length(pca1.mu)+length(pca2.mu)+length(pca3.mu)+length(pca4.mu)+length(pca5.mu))/2;
fisher_vector=zeros(K*2*fisher_dim,num_file);
for i=1:num_file
	
	cur_out_file=[fisher_dir,d(i).name(1:end-16),'_pca',num2str(N_pca),'.fisher'];
	if exist (cur_out_file,'file')~=2
		tic
		cur_file=[feature_dir,d(i).name];
		X_cur=single(read_feature(cur_file));
		X_feat_1=pca1.eigvec'*(X_cur(11:40,:)-repmat(pca1.mu,1,length(X_cur(1,:))));
		X_feat_2=pca2.eigvec'*(X_cur(41:136,:)-repmat(pca2.mu,1,length(X_cur(1,:))));
		X_feat_3=pca3.eigvec'*(X_cur(137:244,:)-repmat(pca3.mu,1,length(X_cur(1,:))));
		X_feat_4=pca4.eigvec'*(X_cur(245:340,:)-repmat(pca4.mu,1,length(X_cur(1,:))));
        X_feat_5=pca5.eigvec'*(X_cur(341:436,:)-repmat(pca5.mu,1,length(X_cur(1,:))));
		    
		X_fisher_1=yael_fisher (X_feat_1, gmm1.w, gmm1.mu, gmm1.sigma, 'sigma','nonorm');
		X_fisher_2=yael_fisher (X_feat_2, gmm2.w, gmm2.mu, gmm2.sigma, 'sigma','nonorm');
		X_fisher_3=yael_fisher (X_feat_3, gmm3.w, gmm3.mu, gmm3.sigma, 'sigma','nonorm');
		X_fisher_4=yael_fisher (X_feat_4, gmm4.w, gmm4.mu, gmm4.sigma, 'sigma','nonorm');
		X_fisher_5=yael_fisher (X_feat_5, gmm5.w, gmm4.mu, gmm5.sigma, 'sigma','nonorm');
        
		X_fisher_1_norm=yael_vecs_normalize((sign(X_fisher_1).*(abs(X_fisher_1).^0.5)),2);	 
		X_fisher_2_norm=yael_vecs_normalize((sign(X_fisher_2).*(abs(X_fisher_2).^0.5)),2);	 
		X_fisher_3_norm=yael_vecs_normalize((sign(X_fisher_3).*(abs(X_fisher_3).^0.5)),2);	 
		X_fisher_4_norm=yael_vecs_normalize((sign(X_fisher_4).*(abs(X_fisher_4).^0.5)),2);	 
		X_fisher_5_norm=yael_vecs_normalize((sign(X_fisher_5).*(abs(X_fisher_5).^0.5)),2);	 
	
        
		X_fisher=[X_fisher_1_norm;X_fisher_2_norm;X_fisher_3_norm;X_fisher_4_norm;X_fisher_5_norm];
		
		fisher_vector(:,i)=X_fisher;
		fid=fopen(cur_out_file,'w');
		fwrite(fid,1,'float');
		fwrite(fid,fisher_dim*K*2,'float');
		fwrite(fid,X_fisher,'float');
		disp(['Finish fisher vector for ',d(i).name])
		fclose(fid);
		toc
	else
		disp([cur_out_file,' exists...'])
		fisher_vector(:,i)=read_feature(cur_out_file);
	end	
end
    
%% Calculate Fisher vector for each testing video 

d=dir([feature_dir,'actioncliptest*.binary']);
num_file=length(d);
fisher_dim=(length(pca1.mu)+length(pca2.mu)+length(pca3.mu)+length(pca4.mu)+length(pca5.mu))/2;
fisher_vector_test=zeros(K*2*fisher_dim,num_file);
for i=1:num_file
	
	cur_out_file=[fisher_dir,d(i).name(1:end-16),'_pca',num2str(N_pca),'.fisher'];
	if exist (cur_out_file,'file')~=2
		tic
		cur_file=[feature_dir,d(i).name];
		X_cur=single(read_feature(cur_file));
		X_feat_1=pca1.eigvec'*(X_cur(11:40,:)-repmat(pca1.mu,1,length(X_cur(1,:))));
		X_feat_2=pca2.eigvec'*(X_cur(41:136,:)-repmat(pca2.mu,1,length(X_cur(1,:))));
		X_feat_3=pca3.eigvec'*(X_cur(137:244,:)-repmat(pca3.mu,1,length(X_cur(1,:))));
		X_feat_4=pca4.eigvec'*(X_cur(245:340,:)-repmat(pca4.mu,1,length(X_cur(1,:))));
        	X_feat_5=pca5.eigvec'*(X_cur(341:436,:)-repmat(pca5.mu,1,length(X_cur(1,:))));
		    
		X_fisher_1=yael_fisher (X_feat_1, gmm1.w, gmm1.mu, gmm1.sigma, 'sigma','nonorm');
		X_fisher_2=yael_fisher (X_feat_2, gmm2.w, gmm2.mu, gmm2.sigma, 'sigma','nonorm');
		X_fisher_3=yael_fisher (X_feat_3, gmm3.w, gmm3.mu, gmm3.sigma, 'sigma','nonorm');
		X_fisher_4=yael_fisher (X_feat_4, gmm4.w, gmm4.mu, gmm4.sigma, 'sigma','nonorm');
		X_fisher_5=yael_fisher (X_feat_5, gmm5.w, gmm4.mu, gmm5.sigma, 'sigma','nonorm');
        
		X_fisher_1_norm=yael_vecs_normalize((sign(X_fisher_1).*(abs(X_fisher_1).^0.5)),2);	 
		X_fisher_2_norm=yael_vecs_normalize((sign(X_fisher_2).*(abs(X_fisher_2).^0.5)),2);	 
		X_fisher_3_norm=yael_vecs_normalize((sign(X_fisher_3).*(abs(X_fisher_3).^0.5)),2);	 
		X_fisher_4_norm=yael_vecs_normalize((sign(X_fisher_4).*(abs(X_fisher_4).^0.5)),2);	 
		X_fisher_5_norm=yael_vecs_normalize((sign(X_fisher_5).*(abs(X_fisher_5).^0.5)),2);	 
	
        
		X_fisher=[X_fisher_1_norm;X_fisher_2_norm;X_fisher_3_norm;X_fisher_4_norm;X_fisher_5_norm];
		
		fisher_vector_test(:,i)=X_fisher;
		fid=fopen(cur_out_file,'w');
		fwrite(fid,1,'float');
		fwrite(fid,fisher_dim*K*2,'float');
		fwrite(fid,X_fisher,'float');
		disp(['Finish fisher vector for ',d(i).name])
		fclose(fid);
		toc
	else
		disp([cur_out_file,' exists...'])
		fisher_vector_test(:,i)=read_feature(cur_out_file);
	end	
end
%% Train Linear SVM    

label_dir='../data/Hollywood2/labels/train_label.mat';  
load (label_dir); 

label_dir='../data/Hollywood2/labels/test_label.mat';  
load (label_dir); 

disp('Training and Testing labels loaded')

M=1;
c=128;
for i=1:12
%    model{i}=svmtrain(train_label(i,:)',sparse(fisher_vector)',['-t 0 -c ',num2str(c)]);
    model{i}=train(train_label(i,:)',sparse(fisher_vector)',['-B 1 -c ',num2str(c)]);
% save([out_dir,'model_c',num2str(c),'_pca_',num2str(N_pca)],'model')
end
    
%% Test SVM    
% load([out_dir,'model_c',num2str(c),'_pca_',num2str(N_pca)])

for i=1:12
%    [cur_prediction,acc,decv]=svmpredict(test_label(i,:)',sparse(fisher_vector_test'),model{i});
    [cur_prediction,acc,decv]=predict(test_label(i,:)',sparse(fisher_vector_test'),model{i});
    predicted_labels(i,:)=cur_prediction';
    ap(i)=avgPrecision(decv,test_label(i,:)');
end

mAP=mean(ap)
