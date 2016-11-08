load('/media/saurabh/String/WORK/DATASETS/NYU2/splits.mat');

for i=1:numel(testNdxs)
    fprintf('================================ Processing %d/%d ================================ \n',i,numel(testNdxs));
    imName = num2str(testNdxs(i));
    imPath = fullfile('/media/saurabh/String/WORK/DATASETS/NYU2/Images/',[imName '.png']);
    initPath = fullfile('/media/saurabh/String/WORK/DATASETS/NYU2/Images/clusterIIDres_NYU/shading/',[imName '_k10_shading.pgm']);
    resPath = fullfile('/media/saurabh/String/WORK/DATASETS/NYU2/Images/clusterIIDres_NYU/clusterRcnnIID/',[imName '.mat']);
    resDir = '/media/saurabh/String/WORK/DATASETS/NYU2/Images/clusterIIDres_NYU/clusterRcnnIID/';
    if exist(resPath,'file');
        continue;
    else
        anyImageRCNN_clusterIID(imPath, initPath, resDir);
    end
    
    
end
