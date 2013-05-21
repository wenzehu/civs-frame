% Learning algorithm for EM mixture clustering by FRAME model
mex -largeArrayDims CgetMAX1.c;

numEMIteration=10;
numCluster=3;
sx = 60;
sy = 60;
isSoftClassification=false;
inPath = './positiveImages';
cachePath = './feature';
resultPath = '/home/wzhu/Dropbox/civs-frame/mixture3';
if ~exist(cachePath,'dir')
    mkdir(cachePath)
end
if ~exist(resultPath,'dir')
    mkdir(resultPath)
    mkdir(fullfile(resultPath,'img'));
end

nOrient = 16;
nTileRow = 12; %nTileRow \times nTileCol defines the number of paralle chains
nTileCol = 12;

lambdaLearningRate = 0.01;
nIter = 5; % the number of iterations for learning lambda
epsilon = 0.01; % step size for the leapfrog
L = 10; % leaps for the leapfrog
numSample = 3; % how many HMC calls for each learning iteration
isLocalNormalize=true; %
isSeparateLocalNormalize=false;
locationShiftLimit=0;
orientShiftLimit=0;

localHalfx=20;
localHalfy=20;
thresholdFactor=0.01;


isSaved=1;
isComputelogZ=1;

%% Step 0: prepare filter, training images,and filter response on images
f1 = MakeFilter(0.7,nOrient);
for i=1:nOrient
    f1_r{i} =real(f1{i});
    f1_i{i} =imag(f1{i});
end

filters = [f1_r f1_i];
numFilter = length(filters);
halfFilterSizes = zeros(size(filters));
for iF = 1:numFilter
    filters{iF} = single(filters{iF});
    halfFilterSizes(iF)=(size(filters{iF},1)-1)/2;
end

imgCell = cell(0);
files = dir(fullfile(inPath,'*.jpg'));
numImage=length(files);
disp(['start filtering']); tic
if ~exist('feature/','dir')
    mkdir('feature/');
end
for iImg = 1:numImage
    copyfile(fullfile(inPath,files(iImg).name),fullfile(resultPath,'img',files(iImg).name));
    img = imread(fullfile(inPath,files(iImg).name));
    img = imresize(img,[sx, sy]);
    img = im2single(rgb2gray(img));
    img = img-mean(img(:));
    img = img/std(img(:));
    
    S1 = cell(numFilter,1);
    M1 = cell(numFilter,1);
    
    % SUM1
    for iFilter = 1:numFilter
        Y = filter2(filters{iFilter},img);
        S1{iFilter} = abs(single(Y));
        M1{iFilter} = zeros(sx,sy,'single');
    end
    
    
    if isLocalNormalize
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Local Normalization
        h1 = halfFilterSizes(1);
        if isSeparateLocalNormalize
            [S1(1:nOrient)]= ...
                LocalNormalize(S1(1:nOrient),[],h1,round(0.6*h1),round(0.6*h1),thresholdFactor);
            
            [S1(nOrient+1: 2*nOrient)]= ...
                LocalNormalize(S1(nOrient+1: 2*nOrient),[],h1,round(0.6*h1),round(0.6*h1),thresholdFactor);
        else
            [S1(1:nOrient),S1(nOrient+1: 2*nOrient)]= ...
                LocalNormalize(S1(1:nOrient),S1(nOrient+1: 2*nOrient),h1,round(0.6*h1),round(0.6*h1),thresholdFactor);
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    end
    
    %MAX1
    CgetMAX1(1,sx,sy,nOrient,locationShiftLimit,orientShiftLimit,1,S1(1:nOrient),M1(1:nOrient));
    CgetMAX1(1,sx,sy,nOrient,locationShiftLimit,orientShiftLimit,1,S1(nOrient+1: 2*nOrient),M1(nOrient+1: 2*nOrient));
    
    
    mapName = fullfile(cachePath,['SUMMAXmap-image' num2str(iImg) '.mat']);
    current_file_name=files(iImg).name;
    save(mapName, 'M1','current_file_name');
end
disp(['filtering time: ' num2str(toc) ' seconds']);

%% Prepare variables for EM
prob = zeros(1, numCluster);
SUM2scoreAll = zeros(numImage, numCluster);
dataWeightAll = rand(numImage,numCluster);
dataWeightAll = dataWeightAll./repmat(sum(dataWeightAll,2),1,numCluster);

if isSoftClassification==false
    [maxs,ind]=max(dataWeightAll,[],2);  % find max in each row
    dataWeightAll=zeros(size(dataWeightAll));  % set zeros
    dataWeightAll(sub2ind(size(dataWeightAll),1:length(ind),ind'))=1;  % hard classification
end

clusters=struct('imageIndex',cell(numCluster,1),'rHat',[],'lambdaF',[],'logZ',[],'sampleImages',[],'mixtureWeight',[]);
    for iImg=1:size(dataWeightAll,1)
      [~, ind]=max(dataWeightAll(iImg,:));
      clusters(ind).imageIndex=[clusters(ind).imageIndex,iImg];
    end
    it = 0;
  ShowAssignment;

%save dataWeightAll dataWeightAll;
%load dataWeightAll; % remove this line if you do not want strict reproducibility

overAllArea = sum((sx-2*halfFilterSizes).*(sy-2*halfFilterSizes));
for iCluster = 1:numCluster
    clusters(iCluster).logZ=log((2*pi))*(overAllArea/2);
    for iFilter = 1:numFilter
        clusters(iCluster).lambdaF{iFilter} = zeros(sx,sy,'single');
    end
    clusters(iCluster).sampleImages = randn(sx*nTileRow,sy*nTileCol,'single');
end


%% EM iteration
for (it = 1 : numEMIteration)
    disp(['M-step of iteration ' num2str(it)]);
    % initialize the folders and aggregate the rHat map;
    for c = 1:numCluster
        savingFolder=fullfile(resultPath,['iteration' num2str(it) ],['cluster' num2str(c) '/']);
        if ~exist(savingFolder)
            mkdir(savingFolder);
        end
        sumWeight = sum(dataWeightAll(:,c));
        clusters(c).mixtureWeight=sumWeight;
        clusters(c).rHat = cell(numFilter,1);
        for iFilter = 1:numFilter
            clusters(c).rHat{iFilter}=zeros(sx,sy,'single');
        end
    end
    for iImg = 1:numImage
        mapName = fullfile(cachePath,['SUMMAXmap-image' num2str(iImg)]);
        featureMap=load(mapName);
        for iCluster = 1:numCluster
            if dataWeightAll(iImg,iCluster)~=0
                for iFilter = 1:numFilter
                    %rHat{iFilter}= rHat{iFilter}+ featureMap.M1{iFilter}.*(dataWeightiImg)*numImage/sumWeight);
                    clusters(iCluster).rHat{iFilter}= clusters(iCluster).rHat{iFilter}+ featureMap.M1{iFilter}.*(dataWeightAll(iImg,iCluster)/clusters(iCluster).mixtureWeight);
                end
            end
        end
    end
    
    tic
    for iCluster= 1:numCluster
         savingFolder=fullfile(resultPath,['iteration' num2str(it) ],['cluster' num2str(iCluster) '/']);
        [lambdaF,currSample,logZ]=FRAMElearnGPUV2(nIter,filters,clusters(iCluster).rHat,clusters(iCluster).sampleImages,clusters(iCluster).lambdaF,clusters(iCluster).logZ,...
            epsilon,L,lambdaLearningRate,numSample, isSaved,savingFolder,isComputelogZ);
        clusters(iCluster).lambdaF=lambdaF;
        clusters(iCluster).logZ = logZ;
        clusters(iCluster).sampleImages=currSample;
    end
    disp(['learning time of FRAME by GPU for the ' num2str(numCluster) ' clusters takes ' num2str(toc) ' seconds']);
    
    %% E-step
    disp(['E-step of iteration ' num2str(it)]);
    for iImg = 1 : numImage
        mapName = fullfile(cachePath,['SUMMAXmap-image' num2str(iImg)]);
        featureMap=load(mapName);
        for iCluster= 1:numCluster
            SUM2score=0;
            for (iFilter = 1:numFilter)
                SUM2score = SUM2score + sum(sum(clusters(iCluster).lambdaF{iFilter}.*featureMap.M1{iFilter}));
            end
            SUM2scoreAll(iImg, iCluster)= SUM2score - clusters(iCluster).logZ;
        end % iCluster
    end
    if isSoftClassification
        for (c = 1 : numCluster)
            prob(c) = sum(dataWeightAll(:, c))/numImage;
        end
        for (c = 1 : numCluster)
            for (img = 1 : numImage)
                dataWeightAll(img, c) = prob(c)/sum(prob(1, :).*exp(SUM2scoreAll(img, :)-SUM2scoreAll(img, c)));
            end
        end
    else
        dataWeightAll=zeros(size(dataWeightAll));  % set zeros
        [maxs,ind]=max(SUM2scoreAll,[],2);  % find max in each row
        dataWeightAll(sub2ind(size(SUM2scoreAll),1:length(ind),ind'))=1;  % hard classification
    end
    
    save(fullfile(resultPath,['iteration' num2str(it)], 'dataWeightAll.mat'),'dataWeightAll');
    save(fullfile(resultPath,['iteration' num2str(it)], 'mixtureModel.mat'),'clusters','numImage','filters');
    
    %%%%% generating html
    for iCluster = 1:numCluster
      clusters(iCluster).imageIndex=[];
    end
    for iImg=1:size(dataWeightAll,1)
      [~, ind]=max(dataWeightAll(iImg,:));
      clusters(ind).imageIndex=[clusters(ind).imageIndex,iImg];
    end
    ShowAssignment;
    
    
end %it

