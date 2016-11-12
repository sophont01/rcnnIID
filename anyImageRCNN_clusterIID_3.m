function anyImageRCNN_clusterIID_3(imPath, resDir, sig_c, sig_i)

fprintf('======= Processing %s =======\n',imPath);
[imDir,imName,~]=fileparts(imPath);
imDir_parts = strsplit(imDir,'/');
 
I = im2uint8(imread(imPath));

if nargin<2
    resDir = fullfile(pwd, 'RESULTS', imDir_parts{end});
    sig_c = 0.0001;
    sig_i = 10; %     sig_i = 0.8;
    sig_n = 0.5;
end

tol = 0 ; % LLE TOLERANCE
knnL = 10; % kNN for LLE
knnG = 2; % kNN for LLE
%% Local Grids

g_size = 30;
o_size = 30;
hn = g_size*floor(size(I,1)/g_size);
wn = g_size*floor(size(I,2)/g_size);
I = imresize(I,[ hn wn ]);
S = RollingGuidanceFilter(im2double(I), 3, 0.1, 4);
N = hn*wn;

n=1;
boxes=[];
for i=1:o_size:size(I,1)-g_size+1
    for j=1:o_size:size(I,2)-g_size+1
        boxes(n,:) = [j , i , j+g_size, i+g_size];
        n = n +1;
    end
end

codeRootDir = pwd;
cd './libs/features/RCNN/'
feat_L = getRCNNFeatures(I, boxes);
cd(codeRootDir);
fMap_L = reshape(feat_L{1},[floor(size(I,1)/g_size), floor(size(I,2)/g_size), size(feat_L{1},2)]);
fMap_L = double(fMap_L);
nNeighbors = getGridLLEMatrix_RCNN(fMap_L, hn, wn, knnL, size(fMap_L,3), g_size, tol);

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
proposal_areas = ((proposals(:,4) - proposals(:,2)) .* (proposals(:,3) - proposals(:,1)));
proposals(proposal_areas>0.9*N | proposal_areas<0.1*N , :) = [];

codeRootDir = pwd;
cd './libs/features/RCNN/'
feat_G = getRCNNFeatures(I, proposals);
cd(codeRootDir);
fMap_G = double(feat_G{1});

var_pad = 2; var_patch = 5;
imsegs = im2superpixels_1(S,imName);
GCfeatures = getGCfeatures(I,imsegs);
normGCfeatures=cell2mat(arrayfun(@(x) normalizeGCFeatures(GCfeatures(x,:)), [1:1:imsegs.nseg], 'UniformOutput', false ))';    % Normalize GC features
normGCmap= densifyFeatures(imsegs,normGCfeatures);  % Dense GC features
proposals = proposals';
varianceOfGCMap = zeros(1,N);
for ii=1:size(normGCmap,3)
    varianceOfGCMap = varianceOfGCMap + var(im2col(padarray(normGCmap(:, :, ii), [var_pad var_pad], 'symmetric'), [var_patch var_patch], 'sliding'));
end
vGCMap = reshape(varianceOfGCMap, [hn wn]);

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
nNeighbors2_s = getRpLLEMatrix_RCNN(fMap_G, r, c, hn, wn, knnG, tol);

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

% Compute propagation weights (matting Laplacian, surface normal)
disp('computing laplacian matrix.');
L_S = getLaplacian1(S, zeros(hn, wn), 0.1^5);

% Compute local reflectance constraint (continuous similarity weight)
C = getChrom(S);
% [consVecCont_s, weightMap_s] = getContinuousConstraintMatrix(C, sig_c, sig_i, S);
[consVecCont_s, weightMap_s] = getContinuousConstraintMatrix(C, sig_c, sig_i, S);


Is=double(mexDenseSIFT(I));
Is=normalizeSiftFeatures(Is);
nweightMap_s = getFeatureConstraintMatrix(Is, sig_n, size(Is,3));

% Optimization
spI = speye(N, N);
mk = zeros(N, 1);
mk(nNeighbors(:, 1)) = 1;
mask = spdiags(mk, 0, N, N); % Subsampling mask for local LLE
mk = zeros(N, 1);
mk(nNeighbors2_s(:, 1)) = 1;
mask2 = spdiags(mk, 0, N, N); % Subsampling mask for non-local LLE

lambdaP = 4 ;
A = lambdaP * WRC + 1 * mask * (spI - LLEGRID) + 1 * mask2 * (spI - LLERP) + 1 * L_S + 0.025 * WSC;
% A = 1 * mask * (spI - LLEGRID) + 1 * mask2 * (spI - LLERP) + 1 * L_S; % + 0.025 * WSC;
b = lambdaP * consVecCont_s;


% A = 4 * (WRC - consVecCont_s) + 1 * mask * (spI - LLEGRID) + 1 * mask2 * (spI - LLERP) + 1 * L_S ;
% b = log(max( sqrt(I.*I), 0.0001));
% b = getConsVec(I);

disp('Optimizing the system...');
newS = pcg(A, b, 1e-3, 5000, [], [], []);
res_s = reshape(exp(newS), [hn wn])/2;
res_r = I ./ repmat(res_s, [1 1 3]) /2; 

% Visualization and Saving Results
if ~exist(resDir,'dir');
    mkdir(resDir);
end
% imwrite([I res_r repmat(res_s, [1 1 3])], fullfile(resDir,[imName '_res.png']));
imwrite([I res_r repmat(res_s, [1 1 3])], fullfile(resDir,[imName '_res.png']));
resPath = fullfile(resDir,[imName '.mat']);
save(resPath,'res_s','res_r','boxes','feat_L','proposals','feat_G');
end
