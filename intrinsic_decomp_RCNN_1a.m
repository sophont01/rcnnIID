function intrinsic_decomp_RCNN_1a(imPath, resDir, tol)
[imDir, imName, ~] = fileparts(imPath);
%% global params
expName = '1a';
if nargin<2
    resDir = fullfile(imDir,'results');
    tol = 0.001;
elseif nargin<3
    tol = 0.001;
end

if ~exist(resDir,'dir')
    mkdir(resDir);
end
if ~exist('sig_c','var')
    sig_c = 0.0001;
end
if ~exist('sig_i','var')
    sig_i = 0.8;
end
if ~exist('sig_n','var')
    sig_n = 0.5;
end
g_size = 12; % grid size for local computation

%% Reading inputs
I = im2double(imread(imPath));
[h,w,d] = size(I);

% computing textureLess image
S = RollingGuidanceFilter(I, 3, 0.1, 4); % Requires double image format

% resizing images for grid multiple
hn = g_size*floor(size(I,1)/g_size);
wn = g_size*floor(size(I,2)/g_size);
I = imresize(I,[ hn wn ]);
S = imresize(S,[ hn wn ]);

Nn = hn*wn;
x = (1:hn)/hn;
xMap = repmat(x,[wn 1])';
y = (1:wn)/wn;
yMap = repmat(y,[hn 1]);

%% LLE Sparse Neighbourhood Constraints

n=1;
boxes=[];
for i=1:size(I,1)/g_size
    for j=1:size(I,2)/g_size
        boxes(n,:) = [i , j , i+g_size, j+g_size];
        n = n +1;
    end
end

codeRootDir = pwd;
cd './libs/features/RCNN/'
feat = getRCNNFeatures(I, [], [], [], [], boxes ,false);
cd(codeRootDir);
fMap = reshape(feat{1},[floor(size(I,1)/g_size), floor(size(I,2)/g_size), size(feat{1},2)]);
fMap = double(fMap);
% fMapXY = cat(3,fMap,xMap);
% fMapXY = cat(3,fMapXY,yMap);
% nNeighbors2_s = getGridLLEMatrixNormal_RCNN_tol(fMapXY, hn, wn, 50, size(fMapXY,3), g_size, 0.000001);  % LOCAL
% nNeighbors2_s = getGridLLEMatrixNormal_RCNN_tol(fMap, hn, wn, 10, size(fMap,3), g_size, 0.000001);  % LOCAL
%nNeighbors2_s = getGridLLEMatrixNormal_RCNN_tol(fMap, hn, wn, 10, size(fMap,3), g_size, 0.001);  % LOCAL
nNeighbors2_s = getGridLLEMatrixNormal_RCNN(fMap, hn, wn, 10, size(fMap,3), 12);
nNeighbors = getGridLLEMatrix_RCNN(fMap, hn, wn, 10, size(fMap,3), 12);

% var_pad = 2; var_patch = 5;
% imsegs = im2superpixels_1(S,imName);
% GCfeatures = getGCfeatures(I,imsegs);
% normGCfeatures=cell2mat(arrayfun(@(x) normalizeGCFeatures(GCfeatures(x,:)), [1:1:imsegs.nseg], 'UniformOutput', false ))';    % Normalize GC features
% normGCmap= densifyFeatures(imsegs,normGCfeatures);  % Dense GC features
%
% % global computation
% varianceOfGCMap=zeros(1,Nn);
% for ii=1:size(normGCmap,3)
%     varianceOfGCMap = varianceOfGCMap + var(im2col(padarray(normGCmap(:, :, ii), [var_pad var_pad], 'symmetric'), [var_patch var_patch], 'sliding'));
% end
% vGCMap = reshape(varianceOfGCMap, [hn wn]);

% global computation
configPath = './libs/rp-master/config/rp_4segs.mat';
config = LoadConfigFile(configPath);
proposals = RP(im2uint8(I), config);
cmins = proposals(:,1);
rmins = proposals(:,2);
cmaxs = proposals(:,3);
rmaxs = proposals(:,4);
h = rmaxs - rmins;
w = cmaxs - cmins;
idx = (h<g_size | w<g_size);
proposals(idx,:) = []; % pruning away small boxes


codeRootDir = pwd;
cd './libs/features/RCNN/'
feat_G = getRCNNFeatures(I, [], [], [], [], proposals ,false);
cd(codeRootDir);
fMap_G = reshape(feat_G{1},[floor(size(I,1)/g_size), floor(size(I,2)/g_size), size(feat_G{1},2)]);
fMap_G = double(fMap_G);



proposals = proposals';
varOfProposals = zeros(1,size(proposals,2));
r = []; c = [];
for ii=1:size(proposals,2)
    box = proposals(:,ii);
    boxVMap = vGCMap(box(2):box(4),box(1):box(3));
    tempMin =  min(min(boxVMap));
    [rr,cc] = find(boxVMap==tempMin); % <------
    ridx = randi(numel(rr));
    r(ii) = rr(ridx);
    c(ii) = cc(ridx);
end

r = (r - ones(size(r)))';
c = (c - ones(size(c)))';
proposals = proposals-ones(size(proposals)); % C++ indexing for mex
% nNeighbors = getGridLLEMatrixFeatures_rp_tol(normGCmap, r, c, proposals, 10, size(normGCmap,3), tol);
nNeighbors = getGridLLEMatrixFeatures_rp_tol(normGCmap, r, c, proposals, 50, size(feat_G{1},2), tol);

% % local computation
% varianceOfFMap=zeros(1,Nn);
% for ii=1:size(fMapXY,3)
%     varianceOfFMap = varianceOfFMap + var(im2col(padarray(fMapXY(:, :, ii), [var_pad var_pad], 'symmetric'), [var_patch var_patch], 'sliding'));
% end
% vMap = reshape(varianceOfFMap, [hn wn]);
% mFMapXY_p = permute(vMap, [2 1 3]);
% nNeighbors2_s = getGridLLEMatrixFeatures_Local_tol(fMapXY, mFMapXY_p, 50, size(fMapXY,3), g_size, tol); % Local neighbourhoods constraints

%% Dense Constraints
C = getChrom(S);
thres = 0.001;
nthres = 0.001;
wr = ones(hn, wn);
ws = 1.0;
[consVec, thresMap] = getConstraintsMatrix(C, wr, ws, thres, 0, S, nthres, S);

% Compute propagation weights (matting Laplacian, surface normal)
disp('computing laplacian matrix.');
L_S = getLaplacian1(S, zeros(hn, wn), 0.1^5);
% nweightMap = getFeatureConstraintMatrix(fMap, sig_n, size(fMap,3));
nweightMap = getFeatureConstraintMatrix(fMapXY, sig_n, size(fMapXY,3));

% Compute local reflectance constraint (continuous similarity weight)
[consVecCont_s, weightMap_s] = getContinuousConstraintMatrix(C, sig_c, sig_i, S);

%% Optimization
spI = speye(Nn, Nn);
mk = zeros(Nn, 1);
mk(nNeighbors(:, 1)) = 1;
mask = spdiags(mk, 0, Nn, Nn); % Subsampling mask for local LLE
mk = zeros(Nn, 1);
mk(nNeighbors2_s(:, 1)) = 1;
mask2 = spdiags(mk, 0, Nn, Nn); % Subsampling mask for non-local LLE

% A = 4 * WRC + 0.1 * mask * (spI - LLELOCAL) + 0.05 * mask2 * (spI - LLEGLOBAL) + 1 * L_S + 0.025 * WSC;
A = 4 * WRC + 0.1 * mask * (spI - LLELOCAL) + 1 * mask2 * (spI - LLEGLOBAL) + 1 * L_S + 0.025 * WSC;
b = 4 * consVecCont_s;

disp('Optimizing the system...');
newS = pcg(A, b, 1e-3, 5000, [], []);

%% Visualization and Saving Results
res_s = reshape(exp(newS), [hn wn])/2;
res_r = I ./ repmat(res_s, [1 1 3]) /2;

resPath = fullfile(resDir,[imName '_' sprintf('%f',gamma3) '_' sprintf('%f',tol) '_'  expName '.mat']);
imwrite([I res_r repmat(res_s,[1 1 3])], fullfile(resDir,[imName '_' sprintf('%f',gamma3) '_' sprintf('%f',tol) '_' expName '.png']));
save(resPath,'res_s','res_r','fMap');

end
