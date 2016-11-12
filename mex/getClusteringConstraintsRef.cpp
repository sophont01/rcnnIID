#include "mexBase.h"
#define MIN(X, Y) (X > Y ? Y : X)
#define MAX(X, Y) (X > Y ? X : Y)
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
	//double *Ic_r = mxGetPr(prhs[0]);	// not used
	//double *Ic_g = mxGetPr(prhs[1]);
	//double *Ic_b = mxGetPr(prhs[2]);
	double *Lc = mxGetPr(prhs[0]);
	//int numClus = int(mxGetPr(prhs[4])[0]);
	double *C = mxGetPr(prhs[1]);
	//double *Rp1 = mxGetPr(prhs[6]);
	//double *Rp2 = mxGetPr(prhs[7]);

	int nx[] = {0, 0, 1, -1, -1, 1, 1, -1};
	int ny[] = {1, -1, 0, 0, -1, 1, -1, 1};

	int h = mxGetM(prhs[1]);	// img height
	int w = mxGetN(prhs[1]);	// img width
	//int numClusRefPairs = mxGetM(prhs[6]); // number of cluster reflectance Pairs

	plhs[0] = mxCreateDoubleMatrix(h*w, 1, mxREAL);
	int dims[2] = {h*w, h*w};
	CvSparseMat* m_refConsMat = cvCreateSparseMat(2, dims, CV_32FC1);
	
	for(int j=0;j<h;j++){
		for(int i=0;i<w;i++){
			int p = i*h+j;
			int Cp = int(C[p]);
			//double pIc_r = Ic_r[Cp];
			//double pIc_g = Ic_g[Cp];
			//double pIc_b = Ic_b[Cp];
			double pLc = Lc[Cp];

			double lp 	= log(MAX(sqrt(pLc*pLc), 0.0001));
			//double ln_pIc_r = log(MAX(sqrt(pIc_r*pIc_r), 0.0001));
			//double ln_pIc_g = log(MAX(sqrt(pIc_g*pIc_g), 0.0001));
			//double ln_pIc_b = log(MAX(sqrt(pIc_b*pIc_b), 0.0001));

			//for(int jj=0;jj<h;jj++){
			//	for(int ii=0;ii<w;ii++){
			for(int k=0;k<8;k++)
			{

				int qi = i + nx[k];
				int qj = j + ny[k];
				int q = qi*h+qj;
				if(qi < 0 || qj < 0 || qi >= w || qj >= h)
				{continue;}
				int Cq = int(C[q]);
				//double qIc_r = Ic_r[Cq];
				//double qIc_g = Ic_g[Cq];
				//double qIc_b = Ic_b[Cq];
				double qLc = Lc[Cq];

				double lq   = log(MAX(sqrt(qLc*qLc), 0.0001));
				if(Cp!=Cq)
				{
					continue;
				}
				//double ln_qIc_r = log(MAX(sqrt(qIc_r*qIc_r), 0.0001));
				//double ln_qIc_g = log(MAX(sqrt(qIc_g*qIc_g), 0.0001));
				//double ln_qIc_b = log(MAX(sqrt(qIc_b*qIc_b), 0.0001));
		
			/*	bool refPair = false;
				for(int a=0;a<numClusRefPairs;a++)
				{	if(a!=Cp)
					{
					 	continue;
					}
					for(int b=0;b<numClusRefPairs;b++)
					{
						if(a==Cp && b==Cq)
						{
							refPair = true;
							break;
						}
					}
				}	
				if(!refPair)
				{
					continue;
				}
			*/	

				((float*)cvPtr2D(m_refConsMat, p, p))[0] += 1;
				((float*)cvPtr2D(m_refConsMat, q, q))[0] += 1;
				((float*)cvPtr2D(m_refConsMat, p, q))[0] += -1;
				((float*)cvPtr2D(m_refConsMat, q, p))[0] += -1;


				float weight = 2*1; //weight * (exp(-dist*dist/(sig_c*sig_c)));
				//if(k == 2)
				//	mxGetPr(plhs[1])[p] = weight;

				//if(isnan(weight)) weight = 0;
				((float*)cvPtr2D(m_refConsMat, p, p))[0] += weight;
				((float*)cvPtr2D(m_refConsMat, q, q))[0] += weight;
				((float*)cvPtr2D(m_refConsMat, p, q))[0] += -weight;
				((float*)cvPtr2D(m_refConsMat, q, p))[0] += -weight;

				float dI = lp - lq;
				//float dI = lq - lp;
				mxGetPr(plhs[0])[p] += weight * dI;
				mxGetPr(plhs[0])[q] -= weight * dI;


				//}
			//}
			}


		}
	
	}
	pushSparseMatrix(m_refConsMat, "WRClus");
	cvReleaseSparseMat(&m_refConsMat);


}
