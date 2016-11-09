#include "mexBase.h"
#include "LLE_rcnn.h"
#include <math.h>

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
	double *fMap = mxGetPr(prhs[0]); // feature Map
	double *rIdx = mxGetPr(prhs[1]); // vMap rIdx
	double *cIdx = mxGetPr(prhs[2]); // vMap cIdx
	int img_h = mxGetPr(prhs[3])[0]; // imgHeight
	int img_w = mxGetPr(prhs[4])[0]; // imgWidth
	int K = mxGetPr(prhs[5])[0]; // k-NN
	double tol_ = mxGetPr(prhs[6])[0]; // LLE tolerance
	float tol = (float)tol_;
	int Ngrid = mxGetM(prhs[0]);
	int feature_dim = mxGetN(prhs[0]);

	mexPrintf("img_h=%d img_w=%d K=%d tol=%f Ngrid=%d feature_dim=%d\n",img_h, img_w, K, tol, Ngrid, feature_dim);

	
	int *c_pos = new int[Ngrid];
	int *r_pos = new int[Ngrid];	
	int N = img_h*img_w; //imgPixels
	int dims[2] = {N, N};
	CvSparseMat* affinityMatrix = cvCreateSparseMat(2, dims, CV_32FC1);
	cv::Mat1f X(Ngrid, feature_dim);
/*	for(int j=0, n=0; j<ngrid_h; j++){
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
*/
	for(int rp=0;rp<Ngrid; rp++){
		c_pos[rp] = int(rIdx[rp]);
		r_pos[rp] = int(cIdx[rp]);
		for(int k=0; k<feature_dim; k++){
			int idx = k*Ngrid + rp;
			double val = fMap[idx];
			//if(rp < 1){
			//	mexPrintf("%.4f ",val);
			//}
			X(rp,k) = val;
		}
	}


	cv::Mat1f W(Ngrid, K);
	cv::Mat1i neighbors(Ngrid, K);
	//LLE(X, W, neighbors, Ngrid, feature_dim, K);
	LLE(X, W, neighbors, Ngrid, feature_dim, tol, K);

	plhs[0] = mxCreateDoubleMatrix(Ngrid, K+1, mxREAL);
	double *neighborPixels = mxGetPr(plhs[0]);
	for(int n=0;n<Ngrid;n++) {
		int xp = r_pos[n];
		int yp = c_pos[n];
		int p = xp * img_h + yp;
		neighborPixels[n] = p + 1;
		for(int k=0;k<K;k++) {
			if(W(n, k) != 0) {
				float weight = W(n,k);
				if(isnan(weight)) weight = 0;

				int nIdx = neighbors(n, k);
				if(nIdx >= 0) {	
					int xq = r_pos[nIdx];
					int yq = c_pos[nIdx];
					int q = xq * img_h + yq;
					((float*)cvPtr2D(affinityMatrix, p, q))[0] = weight;
					neighborPixels[(k+1)*Ngrid + n] = q + 1;
				}
			}
		}
	}

	pushSparseMatrix(affinityMatrix, "LLERP");
	cvReleaseSparseMat(&affinityMatrix);
	delete [] r_pos;
	delete [] c_pos;
}
