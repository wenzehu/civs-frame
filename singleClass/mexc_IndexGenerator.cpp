/* mex-C: 
 * IndexGenerator.cpp is generating index maps for GibbsSampling in order to avoid repeated computation.
 *	usage: [filterIndex2, lambdaIndex2]= mexc_IndexGenerator(numFilter, size(img,1),size(img,2),filters);
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




void mexFunction(int nlhs, mxArray *plhs[], 
                 int nrhs, const mxArray *prhs[])                
{
   
	const mxArray *f;
	mwSize dimsOutput[2];
	const mxArray *pFI, *pLI, *pfF;
	mwSize dims_filterIndex[3], dims_lambdaIndex[3];
	
	mxArray *pF, *pL ;
    void* start_of_pr;
	int iFilter, cx, cy, halfSize, *xVec, *yVec, from_x, to_x, from_y, to_y, num_xVec, num_yVec, *lambdaFIndex, *filterPosIndex, height_filter,bytes_to_copy;
	int img_x,img_y,numFilter;
	int ind,i,j;
	int ind_filterIndex,ind_lambdaFIndex;
	
	
	numFilter = (int)mxGetScalar(prhs[0]);
 	img_x = (int)mxGetScalar(prhs[1]);
	img_y = (int)mxGetScalar(prhs[2]);
	
    pfF = prhs[3];
  

	dims_filterIndex[0] = img_x;
	dims_filterIndex[1] = img_y;
	dims_filterIndex[2] = numFilter;

		           
	plhs[0] = mxCreateCellArray( 3, dims_filterIndex );


    dims_lambdaIndex[0] = img_x;
	dims_lambdaIndex[1] = img_y;
	dims_lambdaIndex[2] = numFilter;

	plhs[1] = mxCreateCellArray( 3, dims_lambdaIndex );


	for(iFilter=0; iFilter<numFilter; iFilter++){
		for(cx=1; cx<=img_x; cx++){
			for(cy=1; cy<=img_y; cy++){
				
				f = mxGetCell(pfF, iFilter);
				height_filter=mxGetM(f);
				halfSize=(height_filter-1)/2;
			    
				from_x=MAX(1,cx-halfSize);
				to_x=MIN(img_x,cx+halfSize);
				num_xVec=to_x-from_x+1;
				//xVec = new int[num_xVec];
				xVec = (int*) mxCalloc(num_xVec, sizeof(int));
				for(i=0; i<num_xVec; i++)  xVec[i]=from_x+i;
				
				from_y=MAX(1,cy-halfSize);
                to_y=MIN(img_y,cy+halfSize);
				num_yVec=to_y-from_y+1;
                //yVec = new int[num_yVec];
				yVec = (int*) mxCalloc(num_yVec, sizeof(int));
				for(i=0; i<num_yVec; i++)  yVec[i]=from_y+i;
				
				//lambdaFIndex=new int[num_xVec * num_yVec];
				lambdaFIndex = (int*) mxCalloc(num_xVec * num_yVec, sizeof(int));
				ind=0;
				for(i=0;i<num_xVec;i++){
					for(j=0;j<num_yVec;j++){
						lambdaFIndex[ind] = ((xVec[i]-1)+(yVec[j]-1)*img_x)+1;
						ind++;
					}
				}
			

                /////////////////// matlab code
				//[x, y]=meshgrid(xVec,yVec);
                // lambdaFIndex = reshape(sub2ind(size(img),x,y),numel(x),1);
				///////////////////////

               //filterPosIndex=new int[num_xVec * num_yVec];
			   filterPosIndex = (int*) mxCalloc(num_xVec * num_yVec, sizeof(int));
			   ind=0;
			   for(i=0;i<num_xVec;i++){
				   for(j=0;j<num_yVec;j++){
					   filterPosIndex[ind] = ((  (cx-xVec[i]+halfSize+1)  -1)+(  (cy-yVec[j]+halfSize+1)   -1)* height_filter )+1;
					   ind++;
				   }
			   }

			   ///////  matlab code
               //[x, y]=meshgrid(cx-xVec+halfSize+1,cy-yVec+halfSize+1);
               //filterPosIndex =  reshape(sub2ind(size(filters{iFilter}),x,y),numel(x),1);
               ////////////////
			
			  //filterIndex{cx,cy,iFilter}=int32(filterPosIndex);
              //lambdaIndex{cx,cy,iFilter}=int32(lambdaFIndex);
			 

			 ind_filterIndex =  (cx-1) + (cy-1)*dims_filterIndex[0] + iFilter*dims_filterIndex[0]*dims_filterIndex[1];
		     //filterIndex[ind_filterIndex]= filterPosIndex;
			 
			 dimsOutput[0] = num_xVec * num_yVec;
			 dimsOutput[1] = 1;
			 pF = mxCreateNumericArray( 2, dimsOutput, mxINT32_CLASS, mxREAL );
			 start_of_pr = (int*)mxGetData(pF);
             bytes_to_copy = dimsOutput[0] * dimsOutput[1] * mxGetElementSize(pF);
             memcpy( start_of_pr, filterPosIndex, bytes_to_copy );
			 mxSetCell( plhs[0], ind_filterIndex, pF ); 


			 ind_lambdaFIndex = (cx-1) + (cy-1)*dims_lambdaIndex[0] + iFilter*dims_lambdaIndex[0]*dims_lambdaIndex[1];
             //lambdaIndex[ind_lambdaFIndex]= lambdaFIndex;

			 dimsOutput[0] = num_xVec * num_yVec;
			 dimsOutput[1] = 1;
			 pL = mxCreateNumericArray( 2, dimsOutput, mxINT32_CLASS, mxREAL );
			 start_of_pr = (int*)mxGetData(pL);
             bytes_to_copy = dimsOutput[0] * dimsOutput[1] * mxGetElementSize(pL);
             memcpy( start_of_pr, lambdaFIndex, bytes_to_copy );
			 mxSetCell(plhs[1], ind_lambdaFIndex,pL ); 
			 
			}
		}
	}
    
   
}

