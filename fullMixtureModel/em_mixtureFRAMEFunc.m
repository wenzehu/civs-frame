function em_mixtureFRAMEFunc(trainImges,modelPath,cachePath)
% Learning algorithm for EM mixture clustering by FRAME model
%% parameters and path
resultPath = modelPath;
if exist(cachePath,'dir')
rmdir(cachePath,'s');    
end
mkdir(cachePath)
if exist(resultPath,'dir')
   rmdir(resultPath,'s');
end
mkdir(resultPath);
mkdir(fullfile(resultPath,'img'));;

sharedParameters;

% pen for visualization
%pen = vision.ShapeInserter('Shape','Lines','BorderColor','Custom','CustomBorderColor',[0 0 255],'Antialiasing',1);

%% Step 0: prepare filter and training images

f1 = MakeFilter(0.7,nOrient);
for i=1:nOrient
    f1_r{i} =real(f1{i});
    f1_i{i} =imag(f1{i});
end
filters = [f1_r f1_i];
%{
f2 = MakeFilter(0.35,nOrient);
for i=1:nOrient
    f2_r{i} =real(f2{i});
    f2_i{i} =imag(f2{i});
end
filters = [f1_r f1_i f2_r f2_i];
%}
if useDoG
f0 = dog(8,0);
filters = [filters f0];
end

numFilter = length(filters);
halfFilterSizes = zeros(size(filters));
for iF = 1:numFilter
    filters{iF} = single(filters{iF});
    halfFilterSizes(iF)=(size(filters{iF},1)-1)/2;
end
overAllArea = sum((sx-2*halfFilterSizes).*(sy-2*halfFilterSizes));


numImage=length(trainImges);
%%% Generate multi-resolution images
originalResolution = round(numResolution/2);    %3; % original resolution is the one at which the imresize factor = 1, see 11th line beneath this line
allSizex = zeros(1, numResolution);
allSizey = zeros(1, numResolution);
ImageMultiResolution = cell(1, numResolution);
tic
for iImg = 1:numImage
    copyfile(fullfile(trainImges(iImg).path,trainImges(iImg).name),fullfile(resultPath,'img',trainImges(iImg).name));
    disp(['======> start filtering and maxing image ' num2str(iImg)]); tic
    img = imread(fullfile(trainImges(iImg).path,trainImges(iImg).name));
    img = imresize(img,[sx, sy]);
    if size(img,3)==1
        img = gray2rgb(img);
    end
    colorImg = padarray(img,[padding_x padding_y],'replicate');
    img = im2single(rgb2gray(img));
    img = padarray(img,[padding_x padding_y],'replicate');
    img = img-mean(img(:));
    img = img/std(img(:));
    
    % create image pyramid
    scaleFactors= zeros(numResolution,1);
    for(resolution=1:numResolution)
        resizeFactor = 1.0 + (resolution - originalResolution)*scaleStepSize;
        scaleFactors(resolution)=resizeFactor;
        ImageMultiResolution{resolution} = imresize(img, resizeFactor, 'nearest');  % images at multiple resolutions
        [sizex, sizey] = size(ImageMultiResolution{resolution});
        allSizex(resolution) = sizex; allSizey(resolution) = sizey;
    end
    
    % filtering images at multiple resolutions
    SUM1mapFind = applyfilter_MultiResolution(ImageMultiResolution, filters, halfFilterSizes, nOrient,locationShiftLimit,orientShiftLimit,isLocalNormalize,isSeparateLocalNormalize,localHalfx,localHalfy,thresholdFactor);
    mapName = fullfile(cachePath,['SUMMAXmap-image' num2str(iImg) '.mat']);
    save(mapName, 'ImageMultiResolution', 'SUM1mapFind','allSizex', 'allSizey','colorImg','scaleFactors');
    
    disp(['filtering time: ' num2str(toc) ' seconds']);
end

%% Prepare variables for EM
cropedImage=zeros(sx, sy,'single');
SUM1mapLearn = cell(numImage, numFilter);
for img = 1:numImage
    for iFilter = 1:numFilter
        SUM1mapLearn{img, iFilter} = zeros(sx, sy,'single');
    end
end
overAllArea = sum((sx-2*halfFilterSizes).*(sy-2*halfFilterSizes));
clusters=struct('rHat',[],'lambdaF',[],'logZ',[],'sampleImages',[],'mixtureWeight',[]);
for iCluster = 1:numCluster
    clusters(iCluster).logZ=log((2*pi))*(overAllArea/2);
    for iFilter = 1:numFilter
        clusters(iCluster).lambdaF{iFilter} = zeros(sx,sy,'single');
    end
    clusters(iCluster).sampleImages = randn(sx*nTileRow,sy*nTileCol,'single');
end


% intitialization for the first run of M-step
ind=originalResolution;   % original resolution
MrotAll = zeros(numImage, numCluster);    % 0 level
MflipAll = zeros(numImage, numCluster);   % no flip
MindAll = ones(numImage, numCluster) * ind;  % original resolution
MFxAll = ones(numImage, numCluster) * floor(allSizex(ind)/2);  % center x
MFyAll = ones(numImage, numCluster)* floor(allSizey(ind)/2);   % center y
MAX2scoreAll = rand(numImage, numCluster);   % random
%% EM iteration
for it = 1 : numEMIteration
    % to keep the MAX2Score map from previouse iteration
    copy_MAX2scoreAll=MAX2scoreAll;
    clusteredImageIdx = cell(numCluster,1);
    for c = 1:numCluster
        disp(['M-step of iteration ' num2str(it) 'for cluster' num2str(c)]);
        savingFolder=fullfile(resultPath,['iteration' num2str(it) ],['cluster' num2str(c) '/']);
        if ~exist(savingFolder)
            mkdir(savingFolder);
        end
        clusteredImageIdx{c}=[];
        
        %% learning for cluster c
        tic
        t=0; % index of image in the cluster, as well as the number of images in cluster
        for iImg = 1:numImage
            [~, ind]=max(copy_MAX2scoreAll(iImg, :));
            if ind~=c
                continue;
            end
            clusteredImageIdx{c}=[clusteredImageIdx{c},iImg]; % collect the id for each cluster
            % crop the SUM1 maps
            mapName = fullfile(cachePath,['SUMMAXmap-image' num2str(iImg)]);
            load(mapName);    % load ImageMultiResolution, SUM1mapFind, allSizex, allSizey for iImg
            Mrot = MrotAll(iImg, c); Mflip = MflipAll(iImg, c);
            Mind = MindAll(iImg, c); MFx = MFxAll(iImg, c); MFy = MFyAll(iImg, c);
            t = t + 1;  
    	    for sF=0:0
            for (iF = 1: 2) % sine or cosine part
                for (orient = 1 : nOrient)
                    orient1 = orient - Mrot;
                    if (orient1 > nOrient)
                        orient1 = orient1 - nOrient;
                    end
                    if (orient1 <= 0)
                        orient1 = orient1 + nOrient;
                    end
                    Ccopy(SUM1mapLearn{t, orient+(sF*2+iF-1)*nOrient}, SUM1mapFind{Mind, orient1+(sF*2+iF-1)*nOrient}, MFx, MFy, floor(sx/2), floor(sy/2), sx, sy, allSizex(Mind), allSizey(Mind), Mrot*pi/nOrient);
                end
            end
            end
            if useDoG
                  Ccopy(SUM1mapLearn{t,end},SUM1mapFind{Mind,end},MFx,MFy,floor(sx/2),floor(sy/2),sx,sy,allSizex(Mind),allSizey(Mind),Mrot*pi/nOrient);
            end
            % If filp, the orientation of the flipped feature maps will also switch
            if (Mflip > 0)
                SUM1mapLearnTmp = SUM1mapLearn(t, :);
                for sF=0:0
                for (iF = 1: 2)
                    for (iOrient = 1 : nOrient)
                        if (iOrient-1>0)
                            o = nOrient - (iOrient-1) + 1;
                        else
                            o = iOrient;
                        end
                        SUM1mapLearn{t, o+(sF*2+iF-1)*nOrient} = fliplr(SUM1mapLearnTmp{1, iOrient+(sF*2+iF-1)*nOrient});
                    end
                end
                end
                if useDoG
                  SUM1mapLearn{t,end}=fliplr(SUM1mapLearnTmp{1,end});
                end
            end
            
            %cropped the input image
            Ccopy(cropedImage, single(ImageMultiResolution{Mind}), MFx, MFy, floor(sx/2), floor(sy/2), sx, sy, allSizex(Mind), allSizey(Mind), Mrot*pi/nOrient);
            if(Mflip > 0)
                cropedImage=fliplr(cropedImage);
            end
            
            % output cropped image
            gLow = min(cropedImage(:));
            gHigh = max(cropedImage(:));
            img_tem = (cropedImage-gLow)/(gHigh-gLow);
            imwrite(img_tem,fullfile(resultPath,['iteration' num2str(it)],['cropped-cluster-' num2str(c) '-training-' num2str(iImg,'%04d') '.png']));
        end% iImage
        
        % compute mean SUM1 maps
        for iFilter = 1:numFilter
            clusters(c).rHat{iFilter}=zeros(sx, sy,'single');
        end
        for iImg = 1:t
            for iFilter = 1:numFilter
                clusters(c).rHat{iFilter}=clusters(c).rHat{iFilter}+SUM1mapLearn{iImg,iFilter};
            end
        end
        for iFilter = 1:numFilter
            clusters(c).rHat{iFilter}=clusters(c).rHat{iFilter}/t;
        end
        
        % learm the template
        if isWarmStart
            [lambdaF,currSample,logZ]=FRAMElearnGPUV2(nIter,filters,clusters(c).rHat,clusters(c).sampleImages,clusters(c).lambdaF,clusters(c).logZ,...
                epsilon,L,lambdaLearningRate,numSample, isSaved,savingFolder,isComputelogZ);
            clusters(c).lambdaF=lambdaF;
            clusters(c).logZ = logZ;
            clusters(c).sampleImages=currSample;
        else
            lambdaF = cell(numFilters,1)
            for iFilter = 1:numFilters
                lambdaF{iFilter}=zeros(sx,sy,'single');
            end
            logZ = 0;
            sampleImages = randn(sx*nTileRow,sy*nTileCol,'single');
            [lambdaF,currSample,logZ]=FRAMElearnGPUV2(nIter,filters,clusters(c).rHat,sampleImages,lambdaF,logZ,...
                epsilon,L,lambdaLearningRate,numSample, isSaved,savingFolder,isComputelogZ);
        end
        clusters(c).lambdaF=lambdaF;
        clusters(c).logZ = logZ;
        clusters(c).sampleImages=currSample;
        disp(['learning time for cluster ' num2str(c) ' takes ' num2str(toc) ' seconds']);
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %% E step, impute the unknown location, rotation, scale and flip
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        disp(['E-step of iteration ' num2str(it) 'for cluster' num2str(c)]);
        disp(['rotating template of cluster' num2str(c)]);
        alllambda=cell((2*rotateShiftLimit+1)*(1+flipOrNot), numFilter); % the transformed tempaltes (lambdaF)
        for (flip = 0 : flipOrNot)
            if (flip > 0)
                flip_lambdaF=cell(1,numFilter);
                for sF= 0:0 % scale of filter
                for (iF = 1: 2)
                    for iOrient = 1:nOrient
                        if (iOrient-1>0)
                            o = nOrient - (iOrient-1) + 1;
                        else
                            o = iOrient;
                        end
                        flip_lambdaF{o+(sF*2+iF-1)*nOrient} = fliplr(lambdaF{iOrient+(sF*2+iF-1)*nOrient});
                    end
                end
                end
                if useDoG
                     flip_lambdaF{end}=fliplr(lambdaF{end});
		end
                lambdaF=flip_lambdaF;
            end
            for (rot = -rotateShiftLimit : rotateShiftLimit)
                r = rot+rotateShiftLimit+1  + (rotateShiftLimit*2+1)*flip;
                angle=rot*180/nOrient;
		for sF=0:0
                for (iF = 1: 2)
                    for iOrient = 1:nOrient
                        o=iOrient-rot;
                        if(o>nOrient)
                            o=o-nOrient;
                        end
                        if(o<=0)
                            o=o+nOrient;
                        end
                        alllambda{r, o+(sF*2+iF-1)*nOrient}= imrotate(lambdaF{iOrient+(sF*2+iF-1)*nOrient},-angle,'bilinear','loose');
                    end
                end
		end
		if useDoG
                alllambda{r,end}=imrotate(lambdaF{end},-angle,'bilinear','loose');
           	end
	    end
        end
        
        %scores every image by detetion process allowing transformed templates to locally shift
        for (iImg = 1 : numImage)
            tic
            mapName = fullfile(cachePath,['SUMMAXmap-image' num2str(iImg)]);
            load(mapName);
            gpuSUM1mapFind=SUM1mapFind;
            for iEl = 1:numel(SUM1mapFind)
		gpuSUM1mapFind{iEl}=gpuArray(SUM1mapFind{iEl});
	   end
            MMAX2 = -Inf;
            for (flip = 0 : flipOrNot)
                for (rot = -rotateShiftLimit : rotateShiftLimit)
                    r = rot+rotateShiftLimit+1 + (rotateShiftLimit*2+1)*flip;
                    [allFx, allFy,MAX2score] = FRAME_SUM2_LogZV2(numResolution, allSizex, allSizey, numFilter, alllambda(r, :), logZ, gpuSUM1mapFind,halfTemplatex,halfTemplatey,halfFilterSizes(1));
                    [maxOverResolution, ind] = max(MAX2score);   % most likely resolution
                    if (MMAX2 < maxOverResolution)
                        MMAX2 = maxOverResolution; Mrot = rot; Mflip = flip;
                        Mind = ind; MFx = allFx(ind); MFy = allFy(ind);
                    end
                end
            end
            MAX2scoreAll(iImg, c) = MMAX2;
            MrotAll(iImg, c) = Mrot; MflipAll(iImg, c) = Mflip;
            MindAll(iImg, c) = Mind; MFxAll(iImg, c) = MFx; MFyAll(iImg, c) = MFy;           
            
            % draw bounding box
            rect_Vertices=[-halfTemplatex -halfTemplatey; -halfTemplatex halfTemplatey; halfTemplatex halfTemplatey; halfTemplatex -halfTemplatey; -halfTemplatex -halfTemplatey];
            top_left_Vertices=[-halfTemplatex-0.1*halfTemplatex -halfTemplatey; -halfTemplatex-0.1*halfTemplatex -halfTemplatey+0.1*halfTemplatey; -halfTemplatex -halfTemplatey+0.1*halfTemplatey];
            combined_Vertices=[rect_Vertices; top_left_Vertices];
            if Mflip==1
                combined_Vertices(:,2)=-1*combined_Vertices(:,2);
            end
            theta = Mrot*pi/nOrient;
            R=[cos(theta) -sin(theta);sin(theta) cos(theta)];    %clockwise rotate theta
            combined_Vertices=combined_Vertices*R+ones(size(combined_Vertices,1),1)*[MFx, MFy];
            combined_Vertices = [combined_Vertices combined_Vertices([2:end 1],:)];            
            combined_Vertices=combined_Vertices/scaleFactors(Mind);
            %visImg = step(pen,colorImg,int32(combined_Vertices(:,[2 1 4 3])));
            %fileName= fullfile(resultPath,['iteration' num2str(it)], ['cluster-' num2str(c) '-training-' num2str(iImg,'%04d') '.png']);
            %imwrite(visImg,fileName);
            disp(['detection time for image ' num2str(iImg) ' takes ' num2str(toc) ' seconds']);
        end% iImg
    end %c, the cluster index
   
    save(fullfile(resultPath,['model_itr' num2str(it,'%02d') '.mat']),'clusters'); 
    save(fullfile(cachePath,['debug_itr' num2str(it,'%02d') '.mat']));
    %% generating html
    fid = fopen(fullfile(resultPath,['result' num2str(it) '.html']), 'wt');
    for c=1:numCluster
        for i=1:length(clusteredImageIdx{c})
            id=clusteredImageIdx{c}(i);
            img_name = fullfile(['iteration' num2str(it)],[ 'cropped-cluster-' num2str(c) '-training-' num2str(id,'%04d') '.png']);
            fprintf(fid, '%s\n', ['<IMG SRC="' img_name '" height=70 ' 'width=70>']);
        end
        fprintf(fid, '%s\n', ['<br>']);
        fprintf(fid, '%s\n', ['<hr>']);
    end
    
    fprintf(fid, '%s\n', ['<br>']);
    fprintf(fid, '%s\n', ['<hr>']);
    fclose(fid);
end %it
     save(fullfile(resultPath,'model_final.mat'),'clusters','filters');
