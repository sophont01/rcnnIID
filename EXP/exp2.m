addpath(genpath('.'));
param = struct;
CWD = '/home/saurabh/WORK/CODES/intrinsic_texture-master/';
EXP = mfilename;

%% Load data
param.path.dataset.depth = fullfile(CWD,'DATA','Depths');
param.path.dataset.images = fullfile(CWD,'DATA','Images');
param.path.tmp = fullfile(CWD,'tmp');
mfRD=matfile(fullfile(param.path.dataset.depth,'rawDepths.mat'));
mfI=matfile(fullfile(param.path.dataset.images,'images.mat'));

%% Read data
idx=171;  % data id to be loaded from mat files
imRD = double(mfRD.rawDepths(:,:,idx));
im = mfI.images(:,:,:,idx);

%% De-texture image
tic
fprintf('START : Detexturing image %6.2f\n',toc);
imS=regcovsmooth(im,10,6,0.1,'M1');
fprintf('END : Detexturing image %6.2f\n-----------\n',toc);
im = im2double(im);
imS = mat2gray(imS);
imT=double(im)-double(imS);


%% Segment image & compute features
param.path.SP = fullfile(CWD,'/libs/segment/segment');
param.SP.sigma = 0.8;
param.SP.k = 100;
param.SP.min = 100;
% tmp_in = [param.path.tmp 'im.ppm'];
% imwrite(imS,tmp_in);
fprintf('START : Segmenting image & computing features %6.2f\n',toc);
imsegs = im2superpixels(imS);
features = mcmcGetSuperpixelData(im,imsegs);
fprintf('END : Segmenting image & computing features %6.2f\n----------\n',toc);

%% IID
fprintf('START : Computing intrinsic images %6.2f\n',toc);
[reflectance, shading] = intrinsic_decomp(im, imS, imRD, 0.0001, 0.8, 0.5);
fprintf('END : Computing intrinsic images %6.2f\n----------\n',toc);

%% Saving results
param.path.results=fullfile(CWD,'RESULTS');
results.im_idx = idx;
results.imsegs = imsegs;
results.features = features;
results.imS = imS;
results.reflectance = reflectance;
results.shading = shading;
save(fullfile(param.path.results,[EXP '_' num2str(idx) '.mat']),'results');

