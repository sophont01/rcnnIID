load('/media/saurabh/String/WORK/DATASETS/NYU2/splits.mat');
Ipath = '/media/saurabh/String/WORK/DATASETS/NYU2/Images/';
Dpath = '/media/saurabh/String/WORK/DATASETS/NYU2/Depths/';
Apath_r = '/media/saurabh/String/WORK/DATASETS/NYU2/Images/clusterIIDres_NYU/reflectance/';
Apath_s = '/media/saurabh/String/WORK/DATASETS/NYU2/Images/clusterIIDres_NYU/shading/';
Bpath = '/media/saurabh/String/WORK/RESULTS/intrinsic_texture-master/RCNN/withSift_reparam/';
ABpath = '/media/saurabh/String/WORK/DATASETS/NYU2/Images/clusterIIDres_NYU/clusterRcnnIID/';
GTk_path_r = '/media/saurabh/String/WORK/RESULTS/DA_intrinsic/Koltun/albedo/';
GTk_path_s = '/media/saurabh/String/WORK/RESULTS/DA_intrinsic/Koltun/shading/';
i = 1;
while i<numel(testNdxs)
    Iname = num2str(testNdxs(i));
    
    B = load(fullfile(Bpath, sprintf('%06d.mat',testNdxs(i))));
    [wn,hn] = size(B.res_s);
    
    I = imresize(im2double(imread(fullfile(Ipath, [Iname '.png']))),[wn,hn]);
    D = imresize(im2double(imread(fullfile(Dpath, [Iname '.png']))),[wn,hn]);
    
    GTk_r = imresize(im2double(imread(fullfile(GTk_path_r, [Iname '_albedo.ppm']))),[wn,hn]);
    GTk_s = imresize(im2double(imread(fullfile(GTk_path_s, [Iname '_shading.ppm']))),[wn,hn]);
    
    A_r = imresize(im2double(imread(fullfile(Apath_r, [Iname '_k10_reflectance.ppm']))),[wn,hn]);
    A_s = imresize(im2double(imread(fullfile(Apath_s, [Iname '_k10_shading.pgm']))),[wn,hn]);
    
    AB = load(fullfile('/media/saurabh/String/WORK/DATASETS/NYU2/Images/clusterIIDres_NYU/clusterRcnnIID/', [Iname '.mat']));
    AB_r = imresize(AB.res_r,[wn,hn]);
    AB_s = imresize(AB.res_s,[wn,hn]);
    
    res = [I GTk_r A_r B.res_r AB_r; repmat(D,[1 1 3]) GTk_s repmat(A_s,[1 1 3]) repmat(B.res_s,[1 1 3]) repmat(AB_s,[1 1 3])];
    clf;
    axes('visible','on','Units','normal','OuterPosition',[0 0 1 1],'Position',[0 0 1 1]);
    
    imshow(res);
    title(sprintf('%s',Iname));
    
    inS = input('N/P','s');
    if strcmpi(inS, 'P')==1
        if i>2
            i=i-1;
            continue;
        else
            i=1;
            continue;
        end
    else
        if ~isempty( str2num(inS))
            i = str2num(inS);
        else
            i=i+1;
        end
    end
    
end