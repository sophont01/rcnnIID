function anyImageRCNN_clusterIID(imPath, initImgPath, resDir)
%% GC
fprintf('=========== Processing %s ===========\n',imPath);
[imDir,imName,imExt] = fileparts(imPath);
I = im2uint8(imread(imPath));
if ~isempty(initImgPath)
    I_init = im2double(imread(initImgPath));
else
    I_init = [];
end

g_size = 30;
o_size = 30;
hn = g_size*floor(size(I,1)/g_size);
wn = g_size*floor(size(I,2)/g_size);
I = imresize(I,[ hn wn ]);
I_init = imresize(I_init,[ hn wn ]);
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
feat_G = getRCNNFeatures(I, boxes);
feat_L = feat_G;
cd(codeRootDir);


fMap_G = reshape(feat_G{1},[floor(size(I,1)/g_size), floor(size(I,2)/g_size), size(feat_G{1},2)]);
fMap_G = double(fMap_G);

fMap_L = reshape(feat_L{1},[floor(size(I,1)/g_size), floor(size(I,2)/g_size), size(feat_L{1},2)]);
fMap_L = double(fMap_L);

%% LLE
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

% % % % Compute Sub-Sampled non-local LLE Constraint
% % % % nNeighbors2_s = getGridLLEMatrixNormal_RCNN_tol(fMap, hn, wn, 50, size(fMap,3), 12, 0.000001);  % GLOBAL --> use MCG here
% % % % nNeighbors = getGridLLEMatrixNormal_RCNN_tol(fMap, hn, wn, 50, size(fMap,3), 12, 0.000001); % LOCAl --> use grid here
% % % % % nNeighbors = getGridLLEMatrix_RCNN(fMap_L, hn, wn, 10, size(fMap_L,3), g_size); % LOCAl --> use grid here
% % % % nNeighbors = getGridLLEMatrixNormal_RCNN(fMap_L, hn, wn, 10, size(fMap_L,3), g_size); % LOCAl --> use grid here
% % % % nNeighbors = getGridLLEMatrixFeatures_rp_tol(fMap, hn, wn, 10, size(fMap,3), g_size, 0.001); 
% 
% nNeighbors2_s = getGridLLEMatrixNormal_RCNN(fMap_G, hn, wn, 10, size(fMap_G,3), g_size); % GLOBAL --> use MCG here
% nNeighbors = getGridLLEMatrixNormal_RCNN(fMap_L, hn, wn, 10, size(fMap_L,3), g_size); % LOCAl --> use grid here

nNeighbors2_s = getGridLLEMatrixNormal_RCNN(fMap_G, hn, wn, 10, size(fMap_G,3), g_size); % GLOBAL --> use MCG here
nNeighbors = getGridLLEMatrix_RCNN(fMap_L, hn, wn, 10, size(fMap_L,3), g_size); % LOCAl --> use grid here

%% IID
C = getChrom(S);
thres = 0.001;
nthres = 0.001;
wr = ones(hn, wn);
ws = 1.0;

%[consVec, thresMap] = getConstraintsMatrix(C, wr, ws, thres, 0, S, nthres, S);
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
A = 4 * WRC + 0.01 * mask * (spI - LLENORMAL) + 0.01 * mask2 * (spI - LLENORMAL) + 1 * L_S + 0.025 * WSC;
% A = 4 * WRC + 0.01 * mask * (spI - LLENORMAL) + 0.01 * mask2 * (spI - LLEGRID) + 1 * L_S + 0.025 * WSC;

b = 4 * consVecCont_s;
disp('Optimizing the system...');
% newS = pcg(A, b, 1e-3, 10000, [], []);
% Visualization and Saving Results
newS = pcg(A, b, 1e-3, 5000, [], [], []);
% pcg_init = reshape(log(2*I_init), [hn*wn 1]);
% newS = pcg(A, b, 1e-3, 5000, [], [], pcg_init);

res_s = reshape(exp(newS), [hn wn])/2;
res_r = I ./ repmat(res_s, [1 1 3]) /2;

if ~exist(resDir,'dir');
    mkdir(resDir);
end
imwrite([I res_r repmat(res_s,[1 1 3])], fullfile(resDir,[imName '_res.png']));

resPath = fullfile(resDir,[imName '.mat']);
save(resPath,'res_s','res_r');
end
