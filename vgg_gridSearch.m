
load 'sleeping_1_frame_0025'
pcg_init = reshape(log(2*I_init), [hn*wn 1]);
% newS = pcg(A, b, 1e-3, 5000, [], [], pcg_init);

A = 4 * WRC + 10 * mask * (spI - LLEGRID) + 10 * mask2 * (spI - LLENORMAL) + 1 * L_S + 0.025 * WSC; % <-- FOR VGG
newS_10 = pcg(A, b, 1e-3, 5000, [], [], []);
newS_10_p = pcg(A, b, 1e-3, 5000, [], [], pcg_init);

A = 4 * WRC + 5 * mask * (spI - LLEGRID) + 5 * mask2 * (spI - LLENORMAL) + 1 * L_S + 0.025 * WSC; % <-- FOR VGG
newS_5 = pcg(A, b, 1e-3, 5000, [], [], []);
newS_5_p = pcg(A, b, 1e-3, 5000, [], [], pcg_init);

A = 4 * WRC + 1 * mask * (spI - LLEGRID) + 1 * mask2 * (spI - LLENORMAL) + 1 * L_S + 0.025 * WSC; % <-- FOR VGG
newS_1 = pcg(A, b, 1e-3, 5000, [], [], []);
newS_1_p = pcg(A, b, 1e-3, 5000, [], [], pcg_init);

A = 4 * WRC + 0.5 * mask * (spI - LLEGRID) + 0.5 * mask2 * (spI - LLENORMAL) + 1 * L_S + 0.025 * WSC; % <-- FOR VGG
newS_f5 = pcg(A, b, 1e-3, 5000, [], [], []);
newS_f5_p = pcg(A, b, 1e-3, 5000, [], [], pcg_init);

A = 4 * WRC + 0.1 * mask * (spI - LLEGRID) + 0.1 * mask2 * (spI - LLENORMAL) + 1 * L_S + 0.025 * WSC; % <-- FOR VGG
newS_f1 = pcg(A, b, 1e-3, 5000, [], [], []);
newS_f1_p = pcg(A, b, 1e-3, 5000, [], [], pcg_init);

A = 4 * WRC + 0.05 * mask * (spI - LLEGRID) + 0.05 * mask2 * (spI - LLENORMAL) + 1 * L_S + 0.025 * WSC; % <-- FOR VGG
newS_f05 = pcg(A, b, 1e-3, 5000, [], [], []);
newS_f05_p = pcg(A, b, 1e-3, 5000, [], [], pcg_init);

A = 4 * WRC + 0.01 * mask * (spI - LLEGRID) + 0.01 * mask2 * (spI - LLENORMAL) + 1 * L_S + 0.025 * WSC; % <-- FOR VGG
newS_f01 = pcg(A, b, 1e-3, 5000, [], [], []);
newS_f01_p = pcg(A, b, 1e-3, 5000, [], [], pcg_init);

A = 4 * WRC + 0.005 * mask * (spI - LLEGRID) + 0.005 * mask2 * (spI - LLENORMAL) + 1 * L_S + 0.025 * WSC; % <-- FOR VGG
newS_f005 = pcg(A, b, 1e-3, 5000, [], [], []);
newS_f005_p = pcg(A, b, 1e-3, 5000, [], [], pcg_init);

A = 4 * WRC + 0.001 * mask * (spI - LLEGRID) + 0.001 * mask2 * (spI - LLENORMAL) + 1 * L_S + 0.025 * WSC; % <-- FOR VGG
newS_001 = pcg(A, b, 1e-3, 5000, [], [], []);
newS_001_p = pcg(A, b, 1e-3, 5000, [], [], pcg_init);

save('vgg_gridSearch.mat')
