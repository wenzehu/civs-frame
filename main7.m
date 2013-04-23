% main function for the FRAME model for general images
nOrient = 8;
%% Step 0: prepare filter and training images
mex -largeArrayDims CgetMAX1.c;
mex -largeArrayDims mexc_GibbsSampling.cpp;
%mex CgetMAX1.c;
%mex mexc_GibbsSampling.cpp;

f0 = dog(10,0);
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
f = [f0 f1_r f1_i f2_r f2_i f3_r f3_i];

filters = f;
numFilter = length(f);
numSample = 1;

locationShiftLimit=0;
orientShiftLimit=0;

inPath = './positiveImage';
imgCell = cell(0);
files = dir([inPath '/*.jpg']);

for iImg = 1:length(files)
    img = imread(fullfile(inPath,files(iImg).name));
    img = imresize(img,[100, 100]);
    imgCell{iImg}=ceil(im2double(rgb2gray(img))*8);
    %imgCell{iImg}=ceil(im2double((img))*8);
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
    lambdaF{iFilter} = zeros(size(img));%rand(size(img));
    gradientF{iFilter}= zeros(size(img));
end
currSample = randi(8,size(img));
% store the <F, I> of current sample
rSample = cell(numFilter,1);
for iFilter = 1:numFilter
    rSample{iFilter}=filter2(filters{iFilter},currSample);
end


filterIndex = cell([size(img) numFilter]);
lambdaIndex = cell([size(img) numFilter]);
for iFilter = 1:numFilter
    for cx = 1: size(img,1);
        for cy = 1:size(img,2);
            halfSize = (size(filters{iFilter},1)-1)/2;
            xVec = max(1,cx-halfSize):min(size(img,1),cx+halfSize);
            yVec = max(1,cy-halfSize):min(size(img,2),cy+halfSize);
            [x, y]=meshgrid(xVec,yVec);
            lambdaFIndex = reshape(sub2ind(size(img),x,y),numel(x),1);
            [x, y]=meshgrid(cx-xVec+halfSize+1,cy-yVec+halfSize+1);
            filterPosIndex =  reshape(sub2ind(size(filters{iFilter}),x,y),numel(x),1);
            filterIndex{cx,cy,iFilter}=int32(filterPosIndex);
            lambdaIndex{cx,cy,iFilter}=int32(lambdaFIndex);
        end
    end
end



%mex mexc_GibbsSampling.cpp;
SSD=zeros(100,1);

step_width0 = .01;
for iter = 1:100
    disp( [ 'iteration: ' num2str(iter)]);
    tic 
   % for (tt = 1:5)
    rModel=mexc_GibbsSampling(numSample,size(img,1), size(img,2), 8, numFilter, filterIndex, lambdaIndex, rSample, lambdaF, currSample, filters);
   % end
    disp(['one iteration learning time: ' num2str(toc) ' seconds']);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%5   
    % compute gradient and do graidnet ascent
   %step_width = step_width *0.9
   step_width = step_width0 *(iter<50);
    lambdaNorm = 0;
    gradientNorm =0;
    for iFilter = 1:numFilter
        gradientF{iFilter} = rHat{iFilter}-rModel{iFilter};
        aa = gradientF{iFilter}; 
        gradientNorm = gradientNorm + mean(abs(aa(:)));
    end
    SSD(iter)=gradientNorm/numFilter;
    disp([ 'SSD: ' num2str(SSD(iter))]);
    %step_width = 1e-3%gradientNorm;
    for iFilter = 1:numFilter
        lambdaF{iFilter}=lambdaF{iFilter}+ step_width*gradientF{iFilter};
        
        lambdaNorm = lambdaNorm + norm(lambdaF{iFilter});
    end
    %for iFilter = 1:numFilter
    %	lambdaF{iFilter} = lambdaF{iFilter}/lambdaNorm;
    %    end
    % save synthesied image
    imwrite(currSample/8,[ num2str(iter,'%04d') '.png']);
end
figure;
plot(1:100,SSD);

