function [rModel, pSample]=multiChainHMC(numFilter,lambdaF,filters,pCurrSample, stepsize, L,nRow,nCol)
% function to sample from multiple parallel chains.
% nRow and nCol defines the size of the chain grid.
% we assume input variables lambdaF,filters,pCurrentSample are in CPU, and the return value will also be in CPU
% we assume the pCurrSample is already a large canvas containing a grid of nRow by nCol sampled images

%% 0. prepare the enlarged canvas of lambdaF 
pgLambdaF = cell(size(lambdaF)); % p for paralle
for iEl = 1:numel(lambdaF)
    pgLambdaF{iEl}=repmat(gpuArray(lambdaF{iEl}),nRow,nCol);
end
gFilters = cell(size(filters));
for iEl = 1:numel(filters)
   gFilters{iEl}=gpuArray(filters{iEl});
end
pgCurrSample = gpuArray(pCurrSample);
%% 1. correct the bounary condition, and run HMC
pgSample = gpuMultiHMC(numFilter,pgLambdaF,gFilters,pgCurrSample,stepsize,L,nRow,nCol);
%% 2 compute the rModel
[sx sy]=size(lambdaF{1});
rModel = cell(numFilter,1);
gNumChain =gpuArray(nRow*nCol);
for iFilter = 1:numFilter
      pgRSample=abs(filter2(gFilters{iFilter},pgSample));
      gRSample = parallel.gpu.GPUArray.zeros(sx,sy);
      for iRow = 1:nRow
        for iCol = 1:nCol
            gRSample = gRSample + pgRSample((iRow-1)*sx+1:iRow*sx,(iCol-1)*sy+1:iCol*sy);
        end
      end
      gRSample = gRSample/gNumChain;
      rModel{iFilter}=gather(gRSample);
end
%% convert the sampled image from GPU to CPU
pSample = gather(pgSample);
end % end of main function


function [q]=gpuMultiHMC(numFilter,pgLambdaF, gFilters, pgCurrSample, stepsize, L,nRow,nCol)
% The main routine for sampling using HMC.
% We assume input paramters pgLambdaF, gFitlers, pgCurrSample are alreay in GPU memory.
% The returned value is also in GPU memory.

%% The traditional leapFrog steps
q=pgCurrSample;
sigma2 = .1; % sigma^2
p = parallel.gpu.GPUArray.randn(size(q));
p = p*sqrt(sigma2);
current_p=p;
p=p- (stepsize/2) * multiGradU(q, numFilter, gFilters, pgLambdaF);
for i=1:L
  q=q+ stepsize * p/sigma2;
  if(i~=L) 
      p=p-stepsize * multiGradU(q, numFilter, gFilters, pgLambdaF); 
  end
end

p=p-stepsize/2*multiGradU(q, numFilter, gFilters, pgLambdaF);


%% compute the accept probability for each chain separately 
currentUMap = multiUMap(pgCurrSample, numFilter, gFilters, pgLambdaF);
proposedUMap = multiUMap(q,numFilter,gFilters,pgLambdaF);
current_p = current_p.^2; % reuse the variable, to save GPU memory
p = p.^2;
[sx sy]=size(pgCurrSample);
sx = sx/nRow; sy = sy/nCol;  % now sx and sy is the size of each small image
keys = rand(nRow,nCol);
for iRow = 1:nRow
    for iCol = 1:nCol
        sx0 = (iRow-1)*sx+1;
        sx1 = iRow*sx;
        sy0 = (iCol-1)*sy+1;
        sy1 = iCol*sy;
        
        currentU = sum(sum(currentUMap(sx0:sx1,sy0:sy1)));
        currentK = sum(sum(current_p(sx0:sx1,sy0:sy1)))/2/sigma2;
        
        proposedU = sum(sum(proposedUMap(sx0:sx1,sy0:sy1)));
        proposedK = sum(sum(p(sx0:sx1,sy0:sy1)))/2/sigma2;
        
        accProb = exp(currentU-proposedU+currentK-proposedK);
        fprintf('accept ratio for chain (%d,%d) is %f, ',iRow,iCol,accProb);
        if keys(iRow,iCol)<accProb
            fprintf('accepted!\n');
        else
            fprintf('rejected!\n');
            q(sx0:sx1,sy0:sy1)=pgCurrSample(sx0:sx1,sy0:sy1);
        end
    end %iRow
end % iCol
end % gpuMultiHMC
        

function gdUMap = multiGradU(pgCurrImage, numFilter, gFilters, pgLambdaF)
gdUMap = parallel.gpu.GPUArray.zeros(size(pgCurrImage));
for iFilter = 1:numFilter
      pgRSampleMap = filter2(gFilters{iFilter},pgCurrImage);
      pgRSampleMap = conv2(sign(pgRSampleMap).*pgLambdaF{iFilter},gFilters{iFilter},'same');
      gdUMap = gdUMap +pgRSampleMap;
end
gdUMap = -gdUMap;
end
      


function pgUEnergyMap = multiUMap(pgCurrImage, numFilter, gFilters, pgLambdaF)
    pgUEnergyMap = parallel.gpu.GPUArray.zeros(size(pgCurrImage));
    for iFilter = 1:numFilter
        pgRSample = abs(filter2(gFilters{iFilter},pgCurrImage));
        pgUEnergyMap = pgUEnergyMap + pgRSample.*pgLambdaF{iFilter}; 
    end
    pgUEnergyMap = -pgUEnergyMap;
end
