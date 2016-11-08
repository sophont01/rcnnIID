addpath(genpath('.'));
param = struct;

%% Load data
param.path.dataset.depth = '/media/saurabh/String/WORK/DATASETS/NYU2/Depths/';
param.path.dataset.images = '/media/saurabh/String/WORK/DATASETS/NYU2/Images/';
param.path.tmp = '/home/saurabh/WORK/CODES/intrinsic_texture-master/tmp/';
if(~exist('rawDepths','var'))
    load([param.path.dataset.depth 'rawDepths.mat'])
end
if(~exist('images','var'))
    load([param.path.dataset.images 'images.mat'])
end
%% Read data
idx=171;  % data id to be loaded from mat files
imRD = double(rawDepths(:,:,idx));
im = images(:,:,:,idx);

%% Segment image & compute features
param.path.SP = '/home/saurabh/WORK/CODES/intrinsic_texture-master/libs/segment/segment';
param.SP.sigma = 0.8;
param.SP.k = 100;
param.SP.min = 100;
tmp_in = [param.path.tmp 'im.ppm'];
imwrite(im,tmp_in);
imsegs = im2superpixels(im);
features = mcmcGetSuperpixelData(im,imsegs);

%% De-texture image
tic
imS=regcovsmooth(im,10,6,0.1,'M1');
toc
imT=double(im)-double(imS);

%% IID
[reflectance, shading] = intrinsic_decomp(im2double(im), im2double(imS), imRD, 0.0001, 0.8, 0.5);

%% Saving results
results.im_idx = idx;
results.imsegs = imsegs;
results.features = features;
results.imS = imS;
results.reflectance = reflectance;
results.shading = shading;


