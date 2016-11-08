function exp2_abacus(idx)
idx = str2num(idx);
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
% if(~exist('rawDepths','var'))
%     load(fullfile(param.path.dataset.depth,'rawDepths.mat'))
% end
% if(~exist('images','var'))
%     load(fullfile(param.path.dataset.images,'images.mat'))
% end

%% Read data
% idx=171;  % data id to be loaded from mat files
imRD = mfRD.rawDepths(:,:,idx);
im = mfI.images(:,:,:,idx);

%% De-texture image
tic
fprintf('START : Detexturing image %6.2f\n',toc);
imS=regcovsmooth(im,10,6,0.1,'M1');
fprintf('END : Detexturing image %6.2f\n-----------\n',toc);
imT=double(im)-double(imS);

%% Segment image & compute features
param.path.SP = fullfile(CWD,'/libs/segment/segment');
param.SP.sigma = 0.8;
param.SP.k = 100;
param.SP.min = 100;
% tmp_in = [param.path.tmp 'im.ppm'];
% imwrite(imS,tmp_in);
fprintf('START : Segmenting image & computing features %6.2f\n',toc);
imsegs = im2superpixels(mat2gray(imS));
features = mcmcGetSuperpixelData(im,imsegs);
fprintf('END : Segmenting image & computing features %6.2f\n----------\n',toc);

%% IID
fprintf('START : Computing intrinsic images %6.2f',toc);
[reflectance, shading] = intrinsic_decomp(im2double(im), im2double(imS), double(imRD), 0.0001, 0.8, 0.5);
fprintf('END : Computing intrinsic images %6.2f\n----------\n',toc);

%% Saving results
param.path.results=fullfile(CWD,'RESULTS');
results.im_idx = idx;
results.imsegs = imsegs;
results.features = features;
results.imS = imS;
results.reflectance = reflectance;
results.shading = shading;
save(fullfile(param.path.results,[EXP '.mat']),'results');

end