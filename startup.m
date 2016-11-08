addpath(genpath('.'));
addpath(genpath('~/WORK/CODES/intrinsic_texture-master_RCNN/utils/'));
addpath(genpath('~/WORK/CODES/intrinsic_texture-master_RCNN/tempworkspace/'));
addpath(genpath('~/WORK/CODES/intrinsic_texture-master_RCNN/mex/'));
% addpath(genpath('~/WORK/CODES/intrinsic_texture-master_RCNN/libs/features/RCNN'));


useRCNN = true; % flag to enable RCNN or FRCNN
if useRCNN
%% RCNN addpaths
    RCNNpath = '/media/saurabh/String/WORK/CODES/intrinsic_texture-master/rcnn-depth/eccv14-code';
    addpath(genpath(fullfile(RCNNpath,'rcnn')));
    addpath(fullfile(RCNNpath,'nyu-hooks'));
    addpath(fullfile(RCNNpath,'scripts'));
    addpath(fullfile(RCNNpath,'utils'));
    addpath(fullfile(RCNNpath,'caffe','matlab','caffe'));
    addpath(fullfile(RCNNpath,'rgbdutils'));
    addpath(fullfile(RCNNpath,'rgbdutils','imagestack','matlab'));
    addpath(genpath_exclude( fullfile(RCNNpath,'utils'), '.git')); 
% %     startup_mcg;
%     addpath(fullfile(RCNNpath,'mcg'));
%     addpath(mcg_root_dir);
%     addpath(fullfile(mcg_root_dir,'lib'));
%     addpath(fullfile(mcg_root_dir,'scripts_training'));
%     addpath(fullfile(mcg_root_dir,'datasets'));
%     addpath(fullfile(mcg_root_dir,'depth_features'));
%     addpath(genpath(fullfile(mcg_root_dir,'src')));
    fprintf('rcnn_startup done\n');
else
%% FRCNN addpaths
    FRCNNpath = '/media/saurabh/String/WORK/CODES/intrinsic_texture-master_RCNN/faster_rcnn-master';
    addpath(genpath(FRCNNpath));
    addpath(genpath(fullfile(FRCNNpath, 'utils')));
    addpath(genpath(fullfile(FRCNNpath, 'functions')));
    addpath(genpath(fullfile(FRCNNpath, 'bin')));
    addpath(genpath(fullfile(FRCNNpath, 'experiments')));
    addpath(genpath(fullfile(FRCNNpath, 'imdb')));
    addpath(genpath(fullfile(FRCNNpath, 'external', 'caffe', 'matlab')));
    fprintf('faster_rcnn_startup done\n');
end



