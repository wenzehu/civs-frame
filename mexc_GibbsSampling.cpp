/* mex-C: 
 * 
 *	
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "mex.h"        /* the algorithm is connect to matlab */
#include "math.h"
#include "matrix.h"
#define ABS(x) ((x)>0? (x):(-(x)))
#define MAX(x, y) ((x)>(y)? (x):(y))
#define MIN(x, y) ((x)<(y)? (x):(y))


/* variable declaration */
const mxArray *pFI, *pLI, *prS, *plF, *pfF ;

int numSample, img_x, img_y, numValue, numFilter;
const int ** filterIndex, **lambdaIndex;
const mwSize* dims_filterIndex, *dims_lambdaIndex;

const int* filterPosIndex;
const int* lambdaFIndex;

double ** rSample, **lambdaF, ** rSampleImag;
double ** rModel;

//int heightFilterIndex, widthFilterIndex;
//int heightLambdaIndex, widthLambdaIndex;
int height_rSample, width_rSample;
int height_lambdaF, width_lambdaF;
int height_currSample, width_currSample;
int height_filter, width_filter;
double * currSample;
const double ** filters; double ** filtersImag; // real, image

double find_min(double *pa,int n) {
	int i;
	double min = pa[0];
	for(i = 1; i < n;i++){
		if(pa[i] < min) min = pa[i];
	}
	return min;
}


int find_where(double *p,int n, double k){
	// we assume p[] is sorted in ascending order
	int i=0;
	while(p[i]<k){
		i++;
	}
	return i;
}

void sampling()
{
     int iFilter, iVal;
	 int i,j;
	 const mxArray *ff;
	 double sumOfExpState, minEnergy;
     double newEnergy;
	 double* localEnergy =new double[numValue]; 
	 double* cumPState =new double[numValue]; 
	 int v1;
	 double uRand;
	 int cx,cy, iSample;
	 int ind_filterIndex;
	 int ind_lambdaFIndex;
	 int numIndex_filterPos, numIndex_lambdaF;
	 double newRSample, newRSampleImag;
	 double v0;
	 double rnum;

	 rModel = (double**)mxCalloc( numFilter, sizeof(*rModel) );
	 
	 for(i=0;i<numFilter;i++){
	    
		 rModel[i] = (double*)mxCalloc( img_x*img_y, sizeof(**rModel) );
         for(j=0;j<img_x*img_y;j++){
		    rModel[i][j] = 0.0;
	 }
        
	 }

	

	 for(iSample = 1;iSample<=numSample;iSample++){
        
	  for(cx=0; cx<img_x; cx++){
		 for(cy=0; cy<img_y; cy++){


	            for(i=0;i<numValue;i++){
	              localEnergy[i]=0.0;
	            }

	            v0 = currSample[cx+cy*height_currSample];

	            for(iFilter = 0; iFilter< numFilter; iFilter++){
                    
		            ind_filterIndex = cx + cy * dims_filterIndex[0] + iFilter * dims_filterIndex[0] * dims_filterIndex[1];
		            filterPosIndex = filterIndex[ind_filterIndex];
					ff = mxGetCell(pFI, ind_filterIndex);
					numIndex_filterPos = mxGetM(ff);  // get the number of indexes in current filterPosIndex
                    
					ind_lambdaFIndex = cx + cy*dims_lambdaIndex[0] + iFilter*dims_lambdaIndex[0]*dims_lambdaIndex[1];
					lambdaFIndex = lambdaIndex[ind_lambdaFIndex];   // return the cell content as pointer we have saved
                    ff = mxGetCell(pLI, ind_lambdaFIndex);   // return the cell content as mxArray
					numIndex_lambdaF = mxGetM(ff);  // get the number of indexes in current lambdaFIndex
					
					if (numIndex_filterPos !=numIndex_lambdaF) mexErrMsgTxt("for debugging: stop now");

					for( iVal = 0; iVal<numValue; iVal++){
                        
						/////////////////////////////// matlab code //////////////////////////////////////
						//newRSample = filters{iFilter}(filterPosIndex).*(iVal-v0) + rSample{iFilter}(lambdaFIndex);
						//localEnergy(iVal) = localEnergy(iVal)+sum(sum(abs(newRSample).*lambdaF{iFilter}(lambdaFIndex)));
                        /////////////////////////////////////////////////////////

						newEnergy=0;
						for(i=0; i<numIndex_filterPos; i++){
						
						    newRSample = filters[iFilter][filterPosIndex[i]-1]*(iVal+1-v0)+ rSample[iFilter][lambdaFIndex[i]-1]; 
							newRSampleImag = filtersImag[iFilter][filterPosIndex[i]-1]*(iVal+1-v0) + rSampleImag[iFilter][lambdaFIndex[i]-1]; 
							
							
							newEnergy += sqrt( newRSample * newRSample + newRSampleImag * newRSampleImag) * lambdaF[iFilter][lambdaFIndex[i]-1];
						}
						
						localEnergy[iVal]+=newEnergy;

                    }


                 }

				   

	               minEnergy=find_min(localEnergy,numValue);
				   sumOfExpState=0;
				   				   
				   for(i=0; i<numValue; i++){
                      
				      localEnergy[i] = exp(localEnergy[i]-minEnergy);  // exp
					  sumOfExpState+=localEnergy[i];
                   }

				   for(i=0; i<numValue; i++){
					  if (sumOfExpState == 0) mexErrMsgTxt("warning !! divided by 0!!");
				      localEnergy[i] =  1.0*localEnergy[i]/sumOfExpState;  // probability
                      
				   }

				   cumPState[0]= localEnergy[0];
				   for(i=1; i<numValue; i++){
				     cumPState[i]= cumPState[i-1] + localEnergy[i]; 
				   }

				 

				   rnum=rand();
				   uRand = rnum/RAND_MAX;
				   
				  
				   v1 = find_where(cumPState,numValue,uRand)+1;    // v1~[0,7]+1   
				   
				 				   
				   currSample[cx+cy*height_currSample]=v1;
                
				   // update rSample
                iVal = v1;
                for( iFilter = 0; iFilter<numFilter; iFilter++){


					ind_filterIndex = cx + cy*dims_filterIndex[0] + iFilter*dims_filterIndex[0]*dims_filterIndex[1];
		            filterPosIndex = filterIndex[ind_filterIndex];
					ff = mxGetCell(pFI, ind_filterIndex);
					numIndex_filterPos = mxGetM(ff);  // get the number of indexes in current filterPosIndex
                    
					ind_lambdaFIndex = cx + cy*dims_lambdaIndex[0] + iFilter*dims_lambdaIndex[0]*dims_lambdaIndex[1];
					lambdaFIndex = lambdaIndex[ind_lambdaFIndex];   // return the cell content as pointer we have saved
                    ff = mxGetCell(pLI, ind_lambdaFIndex);   // return the cell content as mxArray
					numIndex_lambdaF = mxGetM(ff);  // get the number of indexes in current lambdaFIndex

					for(i=0; i<numIndex_filterPos; i++){
						rSample[iFilter][lambdaFIndex[i]-1] = filters[iFilter][filterPosIndex[i]-1]*(iVal-v0) + rSample[iFilter][lambdaFIndex[i]-1];
										
						rSampleImag[iFilter][lambdaFIndex[i]-1] = filtersImag[iFilter][filterPosIndex[i]-1]*(iVal-v0) + rSampleImag[iFilter][lambdaFIndex[i]-1];
					    
						
					}

               
				}
				

	    }  //end of cy
	 } // end of cx

	    // update the model mean
        
	     for(iFilter = 0; iFilter<numFilter; iFilter++){
            for(i=0; i<img_x*img_y;i++){
			
			   rModel[iFilter][i] = rModel[iFilter][i] + sqrt( rSample[iFilter][i] * rSample[iFilter][i] + rSampleImag[iFilter][i] * rSampleImag[iFilter][i] )/numSample;
			}
		 }


	 }
	
	 delete localEnergy;
	 delete cumPState;
}

/* mex function is used to pass on the pointers and scalars from matlab, 
   so that heavy computation can be done by C, which puts the results into 
   some of the pointers. After that, matlab can then use these results. 
   
   So matlab is very much like a managing platform for organizing the 
   experiments, and mex C is like a work enginee for fast computation. */

void mexFunction(int nlhs, mxArray *plhs[], 
                 int nrhs, const mxArray *prhs[])                
{
    
    mxArray *f;
	
	int nGrid;
	

    int ind, i, x, y, o, dataDim, j, bytes_to_copy, nGaborFilter;
	mxArray *pA;
    mwSize ndim;
    const mwSize* dims;
    mwSize dimsOutput[2];
    void* start_of_pr;
    mxClassID datatype;
    int iT;
	double *term;

	//rModel=mexc_GibbsSampling(numSample,img_x, img_y, numValue, numFilter, filterIndex, lambdaIndex, rSample, lambdaF);
	/*
	 * input variable 0-4
	 */
	numSample = (int)mxGetScalar(prhs[0]);
 	img_x = (int)mxGetScalar(prhs[1]);
	img_y = (int)mxGetScalar(prhs[2]);
	numValue = (int)mxGetScalar(prhs[3]);
	numFilter = (int)mxGetScalar(prhs[4]);

	
    /*
	 * input variable 5: filterIndex
	 */
    pFI = prhs[5];
    dims_filterIndex = mxGetDimensions(pFI);
    nGrid = dims_filterIndex[0] * dims_filterIndex[1] * dims_filterIndex[2];
    
 
    filterIndex = (const int**)mxCalloc( nGrid, sizeof(*filterIndex) );   
    for (i=0; i<nGrid; ++i)
    {
        f = mxGetCell(pFI, i);
        datatype = mxGetClassID(f);
        if (datatype != mxINT32_CLASS)
            mexErrMsgTxt("warning !! int32 required.");
        filterIndex[i] = (const int*)mxGetPr(f);    /* get the pointer to cell content */
        //heightFilterIndex = mxGetM(f);    /* overwriting is ok, since it is constant */
        //widthFilterIndex = mxGetN(f);
		/*if (widthFilterIndex != 1)
            mexErrMsgTxt("warning !! one cell in filterIndex is not column vector.");*/
    }


	/*
	 * input variable 6: lambdaIndex
	 */
    pLI = prhs[6];
    dims_lambdaIndex = mxGetDimensions(pLI);
    nGrid = dims_lambdaIndex[0] * dims_lambdaIndex[1] * dims_lambdaIndex[2];
    
 
    lambdaIndex = (const int**)mxCalloc( nGrid, sizeof(*lambdaIndex) );  
    for (i=0; i<nGrid; ++i)
    {
        f = mxGetCell(pLI, i);
        datatype = mxGetClassID(f);
        if (datatype != mxINT32_CLASS)
            mexErrMsgTxt("warning !! int32 required.");
        lambdaIndex[i] = (const int*)mxGetPr(f);    /* get the pointer to cell content */
        //heightLambdaIndex = mxGetM(f);    /* overwriting is ok, since it is constant */
        //widthLambdaIndex = mxGetN(f);
		/*if (widthLambdaIndex != 1)
            mexErrMsgTxt("warning !! one cell in lambdaIndex is not column vector.");*/
    }

	

	/*
	 * input variable 7: rSample (complex format)
	 */
    prS = prhs[7];
    dims = mxGetDimensions(prS);
    nGrid = dims[0] * dims[1];
    
 
    rSample = (double**)mxCalloc( nGrid, sizeof(*rSample) );  
	rSampleImag = (double**)mxCalloc( nGrid, sizeof(*rSampleImag) );  
    for (i=0; i<nGrid; ++i)
    {
        f = mxGetCell(prS, i);
        datatype = mxGetClassID(f);
        if (datatype != mxDOUBLE_CLASS)
            mexErrMsgTxt("warning !! double precision required.");
        
		height_rSample = mxGetM(f);    /* overwriting is ok, since it is constant */
        width_rSample = mxGetN(f);
		
		if (mxIsComplex(f)){
            		rSample[i] = (double*)mxGetPr(f);    /* get the pointer to cell content */
		            rSampleImag[i]= (double*)mxGetPi(f);
		}else{
			rSample[i] = (double*)mxGetPr(f);
			term = (double*)mxCalloc( height_rSample * width_rSample, sizeof(double));
			mxSetPi(f,term);
			rSampleImag[i]= (double*)mxGetPi(f);
			
			//rSampleImag[i]= (double*)mxCalloc( height_rSample * width_rSample, sizeof(double));
		}
        
		
    }

	

	/*
	 * input variable 8: lambdaF
	 */
    plF = prhs[8];
    dims = mxGetDimensions(plF);
    nGrid = dims[0] * dims[1];
    
 
    lambdaF = (double**)mxCalloc( nGrid, sizeof(*lambdaF) );   
    for (i=0; i<nGrid; ++i)
    {
        f = mxGetCell(plF, i);
        datatype = mxGetClassID(f);
        if (datatype != mxDOUBLE_CLASS)
            mexErrMsgTxt("warning !! double precision required.");
        lambdaF[i] = (double*)mxGetPr(f);    /* get the pointer to cell content */
        height_lambdaF = mxGetM(f);    /* overwriting is ok, since it is constant */
        width_lambdaF = mxGetN(f);
		
    }


	/*
	 * input variable 9: currSample
	 */
   
	datatype = mxGetClassID(prhs[9]);
    if (datatype != mxDOUBLE_CLASS)
            mexErrMsgTxt("warning !! double precision required.");
    currSample=(double*)mxGetPr(prhs[9]);
	height_currSample = mxGetM(prhs[9]);    /* overwriting is ok, since it is constant */
    width_currSample = mxGetN(prhs[9]);

    /*
	 * input variable 10: filters (complex format)
	 */
    pfF = prhs[10];
	
    dims = mxGetDimensions(pfF);
    nGrid = dims[0] * dims[1];
    
 
    filters = (const double**)mxCalloc( nGrid, sizeof(*filters) );   /* filters */
    filtersImag = (double**)mxCalloc( nGrid, sizeof(*filtersImag) );   /* filters */

	for (i=0; i<nGrid; ++i)
    {
        f = mxGetCell(pfF, i);
		

        datatype = mxGetClassID(f);
        if (datatype != mxDOUBLE_CLASS)
            mexErrMsgTxt("warning !! double precision required.");
        
		height_filter = mxGetM(f);    /* overwriting is ok, since it is constant */
        width_filter = mxGetN(f);
		
		if (mxIsComplex(f)){
           
		    filters[i] = (const double*)mxGetPr(f);    /* get the pointer to cell content */
            filtersImag[i] = (double*)mxGetPi(f);
		}else{
			filters[i] = (const double*)mxGetPr(f);    /* get the pointer to cell content */
            term= (double*)mxCalloc( height_filter * width_filter, sizeof(double));
			mxSetPi(f,term);
			filtersImag[i]= (double*)mxGetPi(f);
			
			//filtersImag[i] =  (double*)mxCalloc( height_filter * width_filter, sizeof(double)); 
		     
		}
		
		
    }

	
	
	sampling();
    

    /* =============================================
     * Handle output variables.
     * ============================================= 
     */
    /*
     * output variable 0: rModel
     */

    dimsOutput[0] = numFilter; dimsOutput[1] = 1;
	plhs[0] = mxCreateCellArray( 2, dimsOutput );
    dimsOutput[0] = img_x; dimsOutput[1] = img_y;
    for( iT = 0; iT < numFilter; ++iT )
    {
        pA = mxCreateNumericArray( 2, dimsOutput, mxDOUBLE_CLASS, mxREAL );
        
        start_of_pr = (double*)mxGetData(pA);
        bytes_to_copy = dimsOutput[0] * dimsOutput[1] * mxGetElementSize(pA);
        memcpy( start_of_pr, rModel[iT], bytes_to_copy );
        mxSetCell( plhs[0], iT, pA );
    }
}

