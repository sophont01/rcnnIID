clear all
clc;
% rcnnIIDMPI_resList;
% rcnnIIDMPI_resList_pool5;
% rcnnIIDMPI_resList_vgg;
% rcnnIIDMPI_resList_mpimini;

% R_list = {
% '/media/saurabh/String/WORK/DATASETS/MPI-Sintel/testing/market_2/frame_0030_100/frame_0030.mat',
% '/media/saurabh/String/WORK/DATASETS/MPI-Sintel/testing/market_2/frame_0030_10/frame_0030.mat',
% '/media/saurabh/String/WORK/DATASETS/MPI-Sintel/testing/market_2/frame_0030_1f6/frame_0030.mat',
% '/media/saurabh/String/WORK/DATASETS/MPI-Sintel/testing/market_2/frame_0030_20/frame_0030.mat',
% '/media/saurabh/String/WORK/DATASETS/MPI-Sintel/testing/market_2/frame_0030_50/frame_0030.mat',
% '/media/saurabh/String/WORK/DATASETS/MPI-Sintel/testing/market_2/frame_0030_5/frame_0030.mat',
% '/media/saurabh/String/WORK/DATASETS/MPI-Sintel/testing/market_2/frame_0030_f0001/frame_0030.mat',
% '/media/saurabh/String/WORK/DATASETS/MPI-Sintel/testing/market_2/frame_0030_f001/frame_0030.mat',
% '/media/saurabh/String/WORK/DATASETS/MPI-Sintel/testing/market_2/frame_0030_f01/frame_0030.mat',
% '/media/saurabh/String/WORK/DATASETS/MPI-Sintel/testing/market_2/frame_0030_f1/frame_0030.mat',
% '/media/saurabh/String/WORK/DATASETS/MPI-Sintel/testing/market_2/frame_0030_f4/frame_0030.mat',
% '/media/saurabh/String/WORK/DATASETS/MPI-Sintel/testing/market_2/frame_0030_f8/frame_0030.mat'
%  };


R_list = {
'/media/saurabh/String/WORK/RESULTS/intrinsic_texture-master_RCNN/MPI/anyImageRCNN_clusterIID_2/lambdaP_gridSerch/frame_0020_10.mat',
'/media/saurabh/String/WORK/RESULTS/intrinsic_texture-master_RCNN/MPI/anyImageRCNN_clusterIID_2/lambdaP_gridSerch/frame_0020_1.mat',
'/media/saurabh/String/WORK/RESULTS/intrinsic_texture-master_RCNN/MPI/anyImageRCNN_clusterIID_2/lambdaP_gridSerch/frame_0020_2.mat',
'/media/saurabh/String/WORK/RESULTS/intrinsic_texture-master_RCNN/MPI/anyImageRCNN_clusterIID_2/lambdaP_gridSerch/frame_0020_4.mat',
'/media/saurabh/String/WORK/RESULTS/intrinsic_texture-master_RCNN/MPI/anyImageRCNN_clusterIID_2/lambdaP_gridSerch/frame_0020_50.mat',
'/media/saurabh/String/WORK/RESULTS/intrinsic_texture-master_RCNN/MPI/anyImageRCNN_clusterIID_2/lambdaP_gridSerch/frame_0020_6.mat',
'/media/saurabh/String/WORK/RESULTS/intrinsic_texture-master_RCNN/MPI/anyImageRCNN_clusterIID_2/lambdaP_gridSerch/frame_0020_8.mat'
};


mseErrs_r = [];
mseErrs_s = [];
mseErrs_a = [];
lmseErrs_r = [];
lmseErrs_s = [];
lmseErrs_a = [];
ssimErrs_r = [];
ssimErrs_s = [];
ssimErrs_a = [];
files = {};

fprintf('\n=======================================================================================================================================\n ');
for f=1:numel(R_list)
    fPath = R_list{f};
    fprintf(sprintf('\n====== f = %d %s ======\n',f, fPath));
    R = load(fPath);    
       
    fPath_parts =  strsplit(fPath,'/');
    [~,fName,~] = fileparts(fPath_parts{end});
%     G.res_r = im2double(imread(fullfile('/media/saurabh/String/WORK/DATASETS/MPI-Sintel/training/albedo', fPath_parts{end-2}, [fName '.png'])));
%     G.res_s = im2double(imread(fullfile('/media/saurabh/String/WORK/DATASETS/MPI-Sintel/full/training/SHADING', fPath_parts{end-2}, [fName '.png'])));
    
    G.res_r = im2double(imread('/media/saurabh/String/WORK/DATASETS/MPI-Sintel/training/albedo/sleeping_1/frame_0020.png'));
    G.res_s = im2double(imread('/media/saurabh/String/WORK/DATASETS/MPI-Sintel/full/training/SHADING/sleeping_1/frame_0020.png'));
    
    G.res_r = G.res_r(9:end-8,3:end-2,:); % for g_size = 30;
    G.res_s = G.res_s(9:end-8,3:end-2,:);
   
    
    mseErrs_r  = [mseErrs_r   evaluate_one_k(R.res_r, G.res_r)];
    mseErrs_s  = [mseErrs_s   evaluate_one_k(R.res_s, G.res_s)];
    
    lmseErrs_r = [lmseErrs_r  levaluate_one_k(R.res_r, G.res_r)];
    lmseErrs_s = [lmseErrs_s  levaluate_one_k(R.res_s, G.res_s)];
    
    ssimErrs_r = [ssimErrs_r  evaluate_ssim_one_k(R.res_r, G.res_r)];
    ssimErrs_s = [ssimErrs_s  evaluate_ssim_one_k_1D(R.res_s, G.res_s)];
   
    files{f} = strcat(fPath_parts{end-2},'/',fName);
end

dssimErrs_r = (1 - ssimErrs_r)./2;
dssimErrs_s = (1 - ssimErrs_s)./2;

mseErrs_a = (mseErrs_r + mseErrs_s)./2;
lmseErrs_a = (lmseErrs_r + lmseErrs_s)./2;
dssimErrs_a = (dssimErrs_r + dssimErrs_s)./2;

avg_mseErrs_r   = mean(mseErrs_r  ( ~isnan(mseErrs_r)   ));
avg_mseErrs_s   = mean(mseErrs_s  ( ~isnan(mseErrs_s)   ));
avg_mseErrs_a   = mean(mseErrs_a  ( ~isnan(mseErrs_a)   ));
avg_lmseErrs_r  = mean(lmseErrs_r ( ~isnan(lmseErrs_r)  ));
avg_lmseErrs_s  = mean(lmseErrs_s ( ~isnan(lmseErrs_s)  ));
avg_lmseErrs_a  = mean(lmseErrs_a ( ~isnan(lmseErrs_a)  ));
avg_dssimErrs_r = mean(dssimErrs_r( ~isnan(dssimErrs_r) ));
avg_dssimErrs_s = mean(dssimErrs_s( ~isnan(dssimErrs_s) ));
avg_dssimErrs_a = mean(dssimErrs_a( ~isnan(dssimErrs_a) ));

% save('/media/saurabh/String/WORK/RESULTS/intrinsic_texture-master_RCNN/MPI/rcnnIIDeval_MPI_test.mat', 'mseErrs_r', 'mseErrs_s', 'mseErrs_a', ...
save('/media/saurabh/String/WORK/RESULTS/intrinsic_texture-master_RCNN/MPI/anyImageRCNN_clusterIID_2/lambdaP_gridSerch/eval.mat', 'mseErrs_r', 'mseErrs_s', 'mseErrs_a', ...
     'lmseErrs_r', 'lmseErrs_s', 'lmseErrs_a', ...
     'dssimErrs_r', 'dssimErrs_s', 'dssimErrs_a', ...
     'avg_mseErrs_r', 'avg_mseErrs_s', 'avg_mseErrs_a',...
     'avg_lmseErrs_r', 'avg_lmseErrs_s', 'avg_lmseErrs_a',...
     'avg_dssimErrs_r', 'avg_dssimErrs_s', 'avg_dssimErrs_a',...
     'R_list');