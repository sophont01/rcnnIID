#include "mexBase.h"
#include "LLE_rcnn.h"
#include <math.h>

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
	double *fMap = mxGetPr(prhs[0]); // normal Map
	int img_h = mxGetPr(prhs[1])[0]; //imgHeight
	int img_w = mxGetPr(prhs[2])[0]; //imgWidth
	int K = mxGetPr(prhs[3])[0]; //k-NN
	int feature_dim = mxGetPr(prhs[4])[0];
	int g_size = mxGetPr(prhs[5])[0]; //gridSize
	
	int N = img_h*img_w; //imgPixels
	int dims[2] = {N, N};
	CvSparseMat* affinityMatrix = cvCreateSparseMat(2, dims, CV_32FC1);
	int ngrid_w = img_w / g_size;
	int ngrid_h = img_h / g_size;
	int Ngrid = ngrid_w * ngrid_h;
	int *x_pos = new int[Ngrid];
	int *y_pos = new int[Ngrid];
	cv::Mat1f X(Ngrid, feature_dim);

	for(int j=0, n=0; j<ngrid_h; j++){
		for(int i=0; i<ngrid_w; i++){
			x_pos[n] = i + floor(g_size/2); // Setting the candidate position to be the mid of the grid
			y_pos[n] = j + floor(g_size/2); //
			for(int k=0; k<feature_dim; k++){
				int idx = k*ngrid_h*ngrid_w + i*ngrid_w + j;	
				X(n,k)=fMap[idx];
			}
			n++;
		}
	}

	cv::Mat1f W(Ngrid, K);
	cv::Mat1i neighbors(Ngrid, K);
	LLE(X, W, neighbors, Ngrid, feature_dim, K);

	plhs[0] = mxCreateDoubleMatrix(Ngrid, K+1, mxREAL);
	double *neighborPixels = mxGetPr(plhs[0]);
	for(int n=0;n<Ngrid;n++) {
		int xp = x_pos[n];
		int yp = y_pos[n];
		int p = xp * img_h + yp;
		neighborPixels[n] = p + 1;
		for(int k=0;k<K;k++) {
			if(W(n, k) != 0) {
				int nIdx = neighbors(n, k);
				if(nIdx >= 0) {
					int xq = x_pos[nIdx];
					int yq = y_pos[nIdx];
					int q = xq * img_h + yq;
					((float*)cvPtr2D(affinityMatrix, p, q))[0] = W(n, k);
					neighborPixels[(k+1)*Ngrid + n] = q + 1;
				}
			}
		}
	}
	pushSparseMatrix(affinityMatrix, "LLEGRID");
	cvReleaseSparseMat(&affinityMatrix);

	delete [] x_pos;
	delete [] y_pos;
}
