% Learning algorithm for EM mixture clustering by FRAME model
mex -largeArrayDims CgetMAX1.c;

numEMIteration=4;
numCluster=3;
sx = 60;
sy = 60;
isSoftClassification=false;
inPath = ['./positiveImages/'];
nOrient = 16;
nTileRow = 12; %nTileRow \times nTileCol defines the number of paralle chains
nTileCol = 12;

lambdaLearningRate = 0.01;
nIter = 15; % the number of iterations for learning lambda
epsilon = 0.01; % step size for the leapfrog
L = 10; % leaps for the leapfrog
numSample = 3; % how many HMC calls for each learning iteration
isLocalNormalize=true; % 
isSeparateLocalNormalize=false;

localHalfx=20;
localHalfy=20;
thresholdFactor=0.01;

%% Step 0: prepare filter and training images

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
overAllArea = sum((sx-2*halfFilterSizes).*(sy-2*halfFilterSizes));
locationShiftLimit=0;
orientShiftLimit=0;

imgCell = cell(0);
files = dir([inPath '/*.jpg']);
numImage=length(files);

disp(['start filtering']); tic
if ~exist('feature/','dir')
       mkdir('feature/');
end
for iImg = 1:numImage
    
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
   
        
    mapName = ['feature/SUMMAXmap' '-image' num2str(iImg)];   
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


    %%%%%%%%%%%%%%%%%%%% show intialization
    cluster=cell(numCluster,1);
    for i=1:size(dataWeightAll,1)
      ind=find(dataWeightAll(i,:)==max(dataWeightAll(i,:)));
      ind=ind(1);
      cluster{ind}=[cluster{ind},i];
    end
    
    %%%%% generating html
    fid = fopen(['result0.html'], 'wt'); 

    for c=1:numCluster
       for i=1:size(cluster{c},2)
           id=cluster{c}(i);
           fprintf(fid, '%s\n', ['<IMG SRC="' 'positiveImages/' files(id).name '" height=70 ' 'width=70>']); 
       end
       fprintf(fid, '%s\n', ['<br>']);
       
       fprintf(fid, '%s\n', ['<hr>']);
    end

    fprintf(fid, '%s\n', ['<br>']);
    fprintf(fid, '%s\n', ['<hr>']);
    fclose(fid);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
%save dataWeightAll dataWeightAll; 
%load dataWeightAll; % remove this line if you do not want strict reproducibility


rHat = cell(numFilter,1);

%% EM iteration
for (it = 1 : numEMIteration)
    
    disp(['M-step of iteration ' num2str(it)]);
    for (c = 1:numCluster)
      
      savingFolder=['results/iteration' num2str(it) '/cluster' num2str(c) '/'];
      if ~exist(savingFolder)
         mkdir(savingFolder);
      end
      tic
      dataWeight = dataWeightAll(:, c) + 0.; 
      sumWeight = sum(dataWeight);
            
      %% learning for cluster c
      
      for iFilter = 1:numFilter
         rHat{iFilter}=zeros(sx,sy);
      end
      
      for iImg = 1:numImage
          mapName = ['feature/SUMMAXmap' '-image' num2str(iImg)];   
          featureMap=load(mapName);    
          
          for iFilter = 1:numFilter
            %rHat{iFilter}= rHat{iFilter}+ featureMap.M1{iFilter}.*(dataWeight(iImg)*numImage/sumWeight);
             rHat{iFilter}= rHat{iFilter}+ featureMap.M1{iFilter}.*(dataWeight(iImg)/sumWeight);
          end
      end
      
      isSaved=1;
      isComputelogZ=1;
      [lambdaF,currSample,logZ]=FRAMElearnGPU(nIter,filters,rHat,numFilter, sx, sy, halfFilterSizes,nTileRow,nTileCol,epsilon,L,lambdaLearningRate,numSample, isSaved,savingFolder,isComputelogZ);
    
      for (img = 1 : numImage)
          
          mapName = ['feature/SUMMAXmap' '-image' num2str(img)];   
          featureMap=load(mapName);    
          
          SUM2score=0;
          
          for (iFilter = 1:numFilter)
             SUM2score = SUM2score + sum(sum(lambdaF{iFilter}.*featureMap.M1{iFilter}));
          end
          SUM2scoreAll(img, c)= SUM2score - logZ;
      end
      
      disp(['learning time of FRAME by GPU for cluster ' num2str(c) ' takes ' num2str(toc) ' seconds']);
              
    end
    %% E-step fractional classification
    disp(['E-step of iteration ' num2str(it)]);
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
    
    save(['results/iteration' num2str(it) '/dataWeightAll.mat'],'dataWeightAll');
    
    
    
    % collect the member ids for each cluster
    cluster=cell(numCluster,1);
    for i=1:size(dataWeightAll,1)
      ind=find(dataWeightAll(i,:)==max(dataWeightAll(i,:)));
      ind=ind(1);
      cluster{ind}=[cluster{ind},i];
    end
    
    %%%%% generating html
    fid = fopen(['result' num2str(it) '.html'], 'wt'); 

    for c=1:numCluster
       for i=1:size(cluster{c},2)
           id=cluster{c}(i);
           fprintf(fid, '%s\n', ['<IMG SRC="' 'positiveImages/' files(id).name '" height=70 ' 'width=70>']); 
       end
       fprintf(fid, '%s\n', ['<br>']);
       
       fprintf(fid, '%s\n', ['<hr>']);
    end

    fprintf(fid, '%s\n', ['<br>']);
    fprintf(fid, '%s\n', ['<hr>']);
    fclose(fid);
                               
   
end %it




