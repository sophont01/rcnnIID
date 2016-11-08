function anyImageRCNN_clusterIID_3(imPath, initImgPath, resDir)

fprintf('======= Processing %s =======\n',imPath);
[imDir,imName,imExt]=fileparts(imPath);
I = im2uint8(imread(imPath));
if ~isempty(initImgPath)
    I_init = im2double(imread(initImgPath));
else
    I_init = [];
end

tol = 0 ;
g_size = 30;
o_size = 30;
hn = g_size*floor(size(I,1)/g_size);
wn = g_size*floor(size(I,2)/g_size);
I = imresize(I,[ hn wn ]);
I_init = imresize(I_init,[ hn wn ]);
S = RollingGuidanceFilter(im2double(I), 3, 0.1, 4);
N = hn*wn;

var_pad = 2; var_patch = 5;
imsegs = im2superpixels_1(S,imName);
GCfeatures = getGCfeatures(I,imsegs);
normGCfeatures = cell2mat(arrayfun(@(x) normalizeGCFeatures(GCfeatures(x,:)), [1:1:imsegs.nseg], 'UniformOutput', false ))';    % Normalize GC features
normGCmap = densifyFeatures(imsegs,normGCfeatures);  % Dense GC features

varianceOfGCMap=zeros(1,N);
for ii=1:size(normGCmap,3)
    varianceOfGCMap = varianceOfGCMap + var(im2col(padarray(normGCmap(:, :, ii), [var_pad var_pad], 'symmetric'), [var_patch var_patch], 'sliding'));
end
vGCMap = reshape(varianceOfGCMap, [hn wn]);


%% Local Grids

n=1;
boxes=[];
for i=1:o_size:size(I,1)-g_size+1
    for j=1:o_size:size(I,2)-g_size+1
        boxes(n,:) = [j , i , min(j+g_size, size(I,2)), min(i+g_size, size(I,1))];
        n = n +1;
    end
end

codeRootDir = pwd;
cd './libs/features/RCNN/'
feat_L = getRCNNFeatures(I, [], [], [], [], boxes ,false);
cd(codeRootDir);
% fMap_L = reshape(feat_L{1},[floor(size(I,1)/g_size), floor(size(I,2)/g_size), size(feat_L{1},2)]);
fMap_L = double(feat_L{1});

r = []; c = [];
boxes = boxes';
for ii=1:size(boxes,2)
    box = boxes(:,ii);
    boxVMap = vGCMap(box(2):box(4),box(1):box(3));
    tempMin =  min(min(boxVMap));
    [rr,cc] = find(boxVMap==tempMin); % <------
    ridx = randi(numel(rr));
    r(ii) = rr(ridx);
    c(ii) = cc(ridx);
end

r = (r - ones(size(r)))';
c = (c - ones(size(c)))';


nNeighbors = getLLEMatrix_RCNN_local(fMap_L, r, c, hn, wn, 50, tol);


%% Global proposals

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
proposals(idx,:) = [];

codeRootDir = pwd;
cd './libs/features/RCNN/'
feat_G = getRCNNFeatures(I, [], [], [], [], proposals, false);
cd(codeRootDir);
fMap_G = double(feat_G{1});

proposals = proposals';
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
nNeighbors2_s = getLLEMatrix_RCNN_global(fMap_G, r, c, hn, wn, 10, tol);


%% IID

I = im2double(I);
S = im2double(S);

if ~exist('sig_c','var')
    sig_c = 0.0001;
end
if ~exist('sig_i','var')
    sig_i = 0.8;
end
if ~exist('sig_n','var')
    sig_n = 0.5;
end

C = getChrom(S);
thres = 0.001;
nthres = 0.001;
wr = ones(hn, wn);
ws = 1.0;

[consVec, thresMap] = getConstraintsMatrix(C, wr, ws, thres, 0, S, nthres, S);
% Compute propagation weights (matting Laplacian, surface normal)
disp('computing laplacian matrix.');
L_S = getLaplacian1(S, zeros(hn, wn), 0.1^5);

Is=double(mexDenseSIFT(I));
Is=normalizeSiftFeatures(Is);
nweightMap_s = getFeatureConstraintMatrix(Is, sig_n, size(Is,3));
% Compute local reflectance constraint (continuous similarity weight)
[consVecCont_s, weightMap_s] = getContinuousConstraintMatrix(C, sig_c, sig_i, S);

% Optimization
spI = speye(N, N);
mk = zeros(N, 1);
mk(nNeighbors(:, 1)) = 1;
mask = spdiags(mk, 0, N, N); % Subsampling mask for local LLE
mk = zeros(N, 1);
mk(nNeighbors2_s(:, 1)) = 1;
mask2 = spdiags(mk, 0, N, N); % Subsampling mask for non-local LLE

% A = 4 * WRC + 1 * mask * (spI - LLEGRID) + 1 * mask2 * (spI - LLENORMAL) + 1 * L_S + 0.025 * WSC;
% A = 4 * WRC + 0.0005 * mask * (spI - LLENORMAL) + 0.0005 * mask2 * (spI - LLENORMAL) + 1 * L_S + 0.025 * WSC;
A = 4 * WRC + 0.0005 * mask * (spI - LLEGRID) + 0.0005 * mask2 * (spI - LLEGLOBAL) + 1 * L_S + 0.025 * WSC;

b = 4 * consVecCont_s;
disp('Optimizing the system...');

% Visualization and Saving Results
% newS = pcg(A, b, 1e-3, 10000, [], []);
% newS = pcg(A, b, 1e-3, 5000, [], [], []);
pcg_init = reshape(log(2*I_init), [hn*wn 1]);
newS = pcg(A, b, 1e-3, 5000, [], [], pcg_init);
% newS = pcg(A, b, 1e-3, 5000, [], [], []);

res_s = reshape(exp(newS), [hn wn])/2;
res_r = I ./ repmat(res_s, [1 1 3]) /2;
imwrite([I res_r repmat(res_s,[1 1 3])], fullfile(resDir,[imName '_res.png']));

resPath = fullfile(resDir,[imName '.mat']);
save(resPath,'res_s','res_r');
end
