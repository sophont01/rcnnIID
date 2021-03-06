clear all; close all;
[numClusters, clusterLabels, Rp1, Rp2] = getIIDClusters('/media/saurabh/String/WORK/DATASETS/MPI-Sintel/testing/bandage_2/frame_0020.png',15);
I = im2double(imread('/media/saurabh/String/WORK/DATASETS/MPI-Sintel/testing/bandage_2/frame_0020.png'));
[L,a,b] = rgb2lab(I);
C = reshape(clusterLabels, [size(I,1) size(I,2)]);


% for i=0:numClusters-1
%     clusterNumPixels(i+1) =  sum(clusterLabels==i);
% end
%
% A = reshape(A, [numClusters numClusters]);
% % Apixel =  A ./ repmat(clusterNumPixels, [numClusters 1]);
% Apixel = reshape(A, [numClusters numClusters]) ./ repmat(clusterNumPixels, [numClusters 1]);
% getClusteringConstraints(I,L,C,A,B);
I1 = I(:,:,1);
I2 = I(:,:,2);
I3 = I(:,:,3);
for i=0:numClusters-1
    cl = find(C==i);
    Ic_r(i+1) = mean(I1(cl));
    Ic_g(i+1) = mean(I2(cl));
    Ic_b(i+1) = mean(I3(cl));
end

for i=0:numClusters-1
    cl = find(C==i);
    Lc(i+1) = mean(L(cl));
end

% wrcl = getClusteringConstraintsRef(Ic_r, Ic_g, Ic_b, Lc, numClusters, C, Rp1, Rp2);