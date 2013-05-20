mex -largeArrayDims CgetMAX1.c;
mex Ctransform.c;
% paramters
nOrient = 16;
nTileRow = 6; %nTileRow \times nTileCol defines the number of paralle chains
nTileCol = 6;
inPath = './positiveImageHummingbird';
outPath = './6by6HummingbirdML';
lambdaLearningRate = 0.01;
sx = 128;
sy = 128;
nIter = 50; % the number of iterations for learning lambda
epsilon = 0.01; % step size for the leapfrog
L = 10; % leaps for the leapfrog
numSample = 3; % how many HMC calls for each learning iteration
isLocalNormalize=true; % 
localHalfx=20;
localHalfy=20;
thresholdFactor=0.01;
% main function for the FRAME model for general images
if ~exist(outPath)
   mkdir(outPath)
end
%% Step 0: prepare filter and training images
mex -largeArrayDims CgetMAX1.c;

f0 = dog(10,0);
f0 = f0;
h0 = (size(f0,1)-1)/2;

f1 = MakeFilter(0.5,nOrient);

for i=1:nOrient
f1_r{i} =real(f1{i});
f1_i{i} =imag(f1{i});
end
h1 = (size(f1{1},1)-1)/2;


f2 = MakeFilter(1,nOrient);
for i=1:nOrient
f2_r{i} =real(f2{i});
f2_i{i} =imag(f2{i});
end


h2 = (size(f2{1},1)-1)/2;

f3 = MakeFilter(0.25,nOrient);

for i=1:nOrient
f3_r{i} =real(f3{i});
f3_i{i} =imag(f3{i});
end


filters = [f0 f1_r f1_i f2_r f2_i f3_r f3_i];
numFilter = length(filters);
halfFilterSizes = zeros(size(filters));
for iF = 1:numFilter
    filters{iF} = single(filters{iF});
    halfFilterSizes(iF)=(size(filters{iF},1)-1)/2;
end
overAllArea = sum((sx-2*halfFilterSizes).*(sy-2*halfFilterSizes));
locationShiftLimit=0;
orientShiftLimit=0;

imgCell = cell(0);
files = dir([inPath '/*.jpg']);
for iImg = 1:length(files)
    img = imread(fullfile(inPath,files(iImg).name));
    img = imresize(img,[sx, sy]);
    img = im2single(rgb2gray(img));
    img = img-mean(img(:));
    img = img/std(img(:));
    imgCell{iImg}=img;
end
%% Step 1: compute training sample averages
rHat = cell(numFilter,1);
for iFilter = 1:numFilter
    rHat{iFilter}=zeros(sx,sy);
end
tic;
for iImg = 1:length(files)
    S1 = cell(numFilter,1);
    M1 = cell(numFilter,1);
    % SUM1
    for iFilter = 1:numFilter
        Y = filter2(filters{iFilter},imgCell{iImg});
        S1{iFilter} = abs(Y);
        M1{iFilter} = zeros(sx,sy,'single');
    end
    if isLocalNormalize
    h0 = halfFilterSizes(1);    
    S1(1)=LocalNormalize(S1(1),[],h0,round(0.6*h0),round(0.6*h0),thresholdFactor);
    
    h1 = halfFilterSizes(2);
    [S1(1+1:1+nOrient),S1(1+nOrient+1: 1+2*nOrient)]= ...
    LocalNormalize(S1(1+1:1+nOrient),S1(1+nOrient+1: 1+2*nOrient),h1,round(0.6*h1),round(0.6*h1),thresholdFactor);
  
    h2 = halfFilterSizes(1+2*nOrient+1);
    [S1(1+2*nOrient+1: 1+3*nOrient), S1(1+3*nOrient+1: 1+4*nOrient)]= ...
    LocalNormalize(S1(1+2*nOrient+1: 1+3*nOrient),S1(1+3*nOrient+1: 1+4*nOrient),h2,round(0.6*h2),round(0.6*h2),thresholdFactor);
    
    h3 = halfFilterSizes(1+4*nOrient+1);  
    [S1(1+4*nOrient+1: 1+5*nOrient), S1(1+5*nOrient+1: 1+6*nOrient) ] = ...
    LocalNormalize(S1(1+4*nOrient+1: 1+5*nOrient), S1(1+5*nOrient+1: 1+6*nOrient) ,h3,round(0.6*h3),round(0.6*h3),thresholdFactor);
    
    %{
    h0 = halfFilterSizes(1);    
    S1(1)=LocalNormalizeV2(S1(1),[],h0,localHalfx,localHalfy,thresholdFactor);
    
    h1 = halfFilterSizes(2);
    S1(1+1:1+nOrient)= LocalNormalizeV2(S1(1+1:1+nOrient),[],h1,localHalfx,localHalfy,thresholdFactor);
    S1(1+nOrient+1: 1+2*nOrient) = LocalNormalizeV2(S1(1+nOrient+1: 1+2*nOrient),[],h1,localHalfx,localHalfy,thresholdFactor);
  
    h2 = halfFilterSizes(1+2*nOrient+1);
    S1(1+2*nOrient+1: 1+3*nOrient) = LocalNormalizeV2(S1(1+2*nOrient+1: 1+3*nOrient),[],h2,localHalfx,localHalfy,thresholdFactor);
    S1(1+3*nOrient+1: 1+4*nOrient) = LocalNormalizeV2(S1(1+3*nOrient+1: 1+4*nOrient),[],h2,localHalfx,localHalfy,thresholdFactor);

    h3 = halfFilterSizes(1+4*nOrient+1);  
    S1(1+4*nOrient+1: 1+5*nOrient) = LocalNormalizeV2(S1(1+4*nOrient+1: 1+5*nOrient),[],h3,localHalfx,localHalfy,thresholdFactor);
    S1(1+5*nOrient+1: 1+6*nOrient) = LocalNormalizeV2(S1(1+5*nOrient+1: 1+6*nOrient),[],h3,localHalfx,localHalfy,thresholdFactor);
    %}

    end
    %MAX1
    M1{1}=S1{1};
    CgetMAX1(1,sx,sy,nOrient,locationShiftLimit,orientShiftLimit,1,S1(1+1:1+nOrient),M1(1+1:1+nOrient));
    CgetMAX1(1,sx,sy,nOrient,locationShiftLimit,orientShiftLimit,1,S1(1+nOrient+1: 1+2*nOrient),M1(1+nOrient+1: 1+2*nOrient));
    
    CgetMAX1(1,sx,sy,nOrient,locationShiftLimit,orientShiftLimit,1,S1(1+2*nOrient+1: 1+3*nOrient),M1(1+2*nOrient+1: 1+3*nOrient));
    CgetMAX1(1,sx,sy,nOrient,locationShiftLimit,orientShiftLimit,1,S1(1+3*nOrient+1: 1+4*nOrient),M1(1+3*nOrient+1: 1+4*nOrient));
    
    CgetMAX1(1,sx,sy,nOrient,locationShiftLimit,orientShiftLimit,1,S1(1+4*nOrient+1: 1+5*nOrient),M1(1+4*nOrient+1: 1+5*nOrient));
    CgetMAX1(1,sx,sy,nOrient,locationShiftLimit,orientShiftLimit,1,S1(1+5*nOrient+1: 1+6*nOrient),M1(1+5*nOrient+1: 1+6*nOrient));
    
    for iFilter = 1:numFilter
        rHat{iFilter}= rHat{iFilter}+M1{iFilter};
    end
end
for iFilter = 1:numFilter
    rHat{iFilter}=rHat{iFilter}/length(files);
end
rHatNorm=0;
for iFilter = 1:numFilter
     h = halfFilterSizes(iFilter);
    rHatNorm = rHatNorm + sum(sum( rHat{iFilter}(h+1:end-h,h+1:end-h)));
end
rHatNorm = rHatNorm/overAllArea;

disp(['finished filtering: '  num2str(toc) ' seconds']);

%% Step 2: optimize the exponential model
% our model parameter
lambdaF = cell(numFilter,1);
gradientF = cell(numFilter,1);
for iFilter = 1:numFilter
    lambdaF{iFilter} = zeros(size(img),'single');
    % the following lines will be useful when lambda is not initialized as zero;
    %{
    h = halfFilterSizes(iFilter);
    lambdaF{iFilter}([1:h,sx-h+1:sx],:)=0;
    lambdaF{iFilter}(:,[1:h,sy-h+1:sy])=0; 
    %}
    gradientF{iFilter}= zeros(size(img),'single');
end
[sx sy]=size(img);
prevSamples = single(randn(sx*nTileRow,sy*nTileCol));
initialLogZ =log((2*pi))*(overAllArea/2);
SSD=zeros(nIter,1);
logZRatioSeries = zeros(nIter,1);
for iter = 1:nIter
    disp( [ 'iteration: ' num2str(iter)]);
    tic 
    [rModel, currSamples]=multiChainHMC_G(numFilter,lambdaF,filters,prevSamples,epsilon,L,numSample,nTileRow,nTileCol);
    % compute z ratio
    logZRatio = computeLogZRatio(prevSamples,currSamples,filters,gradientF,lambdaLearningRate,100);
    logZRatioSeries(iter)=logZRatio;
    %
    disp(['one iteration learning time: ' num2str(toc) ' seconds']);

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%5   
    % compute gradient and do graidnet ascent
    gradientNorm = 0; 
    for iFilter = 1:numFilter
        gradientF{iFilter} = rHat{iFilter}-rModel{iFilter};
        h = halfFilterSizes(iFilter);
        gradientF{iFilter}([1:h,sx-h+1:sx],:)=0;
        gradientF{iFilter}(:,[1:h,sy-h+1:sy])=0;
        aa = gradientF{iFilter}; 
        gradientNorm = gradientNorm + sum(abs(aa(:)));
    end
    SSD(iter)=gradientNorm/overAllArea;
    %lambdaLearningRate = 1e-3%gradientNorm;
    for iFilter = 1:numFilter
        lambdaF{iFilter}=lambdaF{iFilter}+ lambdaLearningRate*gradientF{iFilter};
    end
    prevSamples = currSamples;
    % visualization
    % remove the bounaries and save image
    img = currSamples;
    %{
    h = 5;
    img(1:h,:)=0; img(:,1:h)=0;
    for iRow = 1:nTileRow-1
       img(iRow*sx-h:iRow*sx+h+1,:)=0;
    end
    for iCol = 1:nTileCol-1
       img(:,iCol*sy-h:iCol*sy+h+1)=0;
    end
    img(end-h:end,:)=0;
    img(:,end-h:end)=0;
    %}
    gLow = min(img(:));
    gHigh = max(img(:));
    disp([ 'min: ' num2str(gLow) ' max: ' num2str(gHigh)]);
    disp([ 'SSD: ' num2str(SSD(iter))]);
    disp([ 'Relative SSD: ' num2str(SSD(iter)/rHatNorm)]);
    img = (img-gLow)/(gHigh-gLow);
    imwrite(img,fullfile(outPath,[ num2str(iter,'%04d') '.png']));
end
% re-estimate logz
currLogZ = initialLogZ + sum(logZRatioSeries);
logZ = currLogZ;
save([outPath,'/model.mat'],'lambdaF','filters','currSamples','logZ');
disp([' Final LogZ: ' num2str(currLogZ)]);
figure;
plot(1:nIter,SSD);
saveas(gcf,fullfile(outPath,'SSD.png'));
saveas(gcf,fullfile(outPath,'SSD.pdf'));
figure;
plot(1:nIter,SSD/rHatNorm);
saveas(gcf,fullfile(outPath,'RelativeSSD.png'));
saveas(gcf,fullfile(outPath,'RelativeSSD.pdf'));
bar(1:nIter,logZRatioSeries);
title(['Initial Log Z: ' num2str(initialLogZ,'%e') ', final logZ: ' num2str(currLogZ,'%e')]);
saveas(gcf,fullfile(outPath,'logZRatios.png'));
saveas(gcf,fullfile(outPath,'logZRatios.pdf'));
