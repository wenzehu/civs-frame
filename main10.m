% main function for the FRAME model for general images
nOrient = 16;
nTileRow = 8; % nTileRow \times nTileCol defines the number of paralle chains
nTileCol = 8; 
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


%f4 = MakeFilter(1.5,nOrient);
%f = [f0 f1 f2 f3 f4];
%f = [f0 f1 f2 f3];
filters = [f0 f1_r f1_i f2_r f2_i f3_r f3_i];
numFilter = length(filters);
filterSizes = zeros(size(filters));
for iF = 1:numFilter
    filterSizes(iF)=(size(filters{iF},1)-1)/2;
end

locationShiftLimit=0;
orientShiftLimit=0;

inPath = './positiveImage';
imgCell = cell(0);
files = dir([inPath '/*.jpg']);
for iImg = 1:length(files)
    img = imread(fullfile(inPath,files(iImg).name));
    img = imresize(img,[128, 128]);
    imgCell{iImg}=im2double(rgb2gray(img));
end
%% Step 1: compute training sample averages
sx = size(img,1);
sy = size(img,2);
img = rgb2gray(img);
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
        S1{iFilter} = abs(single(Y));
        M1{iFilter} = zeros(sx,sy,'single');
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
        rHat{iFilter}= rHat{iFilter}+double(M1{iFilter});
    end
end
for iFilter = 1:numFilter
    rHat{iFilter}=rHat{iFilter}/length(files);
end
disp(['finished filtering: '  num2str(toc) ' seconds']);

%% Step 2: optimize the exponential model
% our model parameter
lambdaF = cell(numFilter,1);
gradientF = cell(numFilter,1);
for iFilter = 1:numFilter
    lambdaF{iFilter} = zeros(size(img));
    % the following lines will be useful when lambda is not initialized as zero;
    h = filterSizes(iFilter);
    lambdaF{iFilter}([1:h,sx-h+1:sx],:)=0;
    lambdaF{iFilter}(:,[1:h,sy-h+1:sy])=0; 
    gradientF{iFilter}= zeros(size(img));
end
[sx sy]=size(img);
currSamples = zeros(sx*nTileRow,sy*nTileCol);

step_width = .01;
SSD=zeros(500,1);
for iter = 1:500
    disp( [ 'iteration: ' num2str(iter)]);
    tic 
   
    [rModel, currSamples]=multiChainHMC(numFilter,lambdaF,filters,currSamples,0.01,10,nTileRow,nTileCol);
    disp(['one iteration learning time: ' num2str(toc) ' seconds']);

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%5   
    % compute gradient and do graidnet ascent
    %step_width = step_width *0.9;
    gradientNorm = 0; 
    for iFilter = 1:numFilter
        gradientF{iFilter} = rHat{iFilter}-rModel{iFilter};
        h = filterSizes(iFilter);
        gradientF{iFilter}([1:h,sx-h+1:sx],:)=0;
        gradientF{iFilter}(:,[1:h,sy-h+1:sy])=0;
        aa = gradientF{iFilter}; 
        gradientNorm = gradientNorm + sum(abs(aa(:)))/(sx-2*h)/(sy-2*h);
    end
    SSD(iter)=gradientNorm/numFilter;
    %step_width = 1e-3%gradientNorm;
    for iFilter = 1:numFilter
        lambdaF{iFilter}=lambdaF{iFilter}+ step_width*gradientF{iFilter};
    end
    gLow = min(currSamples(:));
    gHigh = max(currSamples(:));
    disp([ 'min: ' num2str(gLow) ' max: ' num2str(gHigh)]);
    disp([ 'SSD: ' num2str(SSD(iter))]);
    img = (currSamples-gLow)/(gHigh-gLow);
    imwrite(img,[ num2str(iter,'%04d') '.png']);
end

figure;
plot(1:500,SSD);
saveas(gcf,'GPUConvergence.png');

