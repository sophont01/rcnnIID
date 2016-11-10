clear  all;
clc;

% rcnnIIDMPI_resList_mpimini;
rcnnIIDMPI_resList_test;


for i=1:numel(R_list)
    anyImageRCNN_clusterIID_2(R_list{i})
end