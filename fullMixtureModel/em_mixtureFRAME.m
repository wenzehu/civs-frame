% Learning algorithm for EM mixture clustering by FRAME model
mex -largeArrayDims CgetMAX1.c;
mex Ccopy.c; % copy around detect location
sx = 100;
sy = 100;

inPath = ['./positiveImages/'];
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

localHalfx=20;
localHalfy=20;
thresholdFactor=0.01;

%% mixture model and unkown resolution, location, orientation, flip
numResolution=3;
scaleStepSize=0.2;
flipOrNot=0;            % template flip Or not
rotateShiftLimit = 3;   % template rotation  from -rotateShiftLimit to rotateShiftLimit, eg. -2:2 if rotateShiftLimit=2
numEMIteration=4;
numCluster=2;



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

halfTemplatex=floor(sx/2);
halfTemplatey=floor(sy/2);

padding_x=round(halfTemplatex/4);
padding_y=round(halfTemplatey/4);

files = dir([inPath '/*.jpg']);
numImage=length(files);

disp(['start filtering']); 
if ~exist('feature/','dir')
       mkdir('feature/');
end

if ~exist('working/','dir')
       mkdir('working/');
end


%%% Generate multi-resolution images
originalResolution = round(numResolution/2);    %3; % original resolution is the one at which the imresize factor = 1, see 11th line beneath this line 
allSizex = zeros(1, numResolution); 
allSizey = zeros(1, numResolution); 
ImageMultiResolution = cell(1, numResolution); 
tic
for iImg = 1:numImage
   
    disp(['======> start filtering and maxing image ' num2str(iImg)]); tic
  
    img = imread(fullfile(inPath,files(iImg).name));
    img = imresize(img,[sx, sy]);
    img = im2single(rgb2gray(img));
    img = padarray(img,[padding_x padding_y],'replicate');
    img = img-mean(img(:));
    img = img/std(img(:));
    
        
    for(resolution=1:numResolution)
       resizeFactor = 1.0 + (resolution - originalResolution)*scaleStepSize; 
       ImageMultiResolution{resolution} = imresize(img, resizeFactor, 'nearest');  % images at multiple resolutions
       [sizex, sizey] = size(ImageMultiResolution{resolution}); 
       allSizex(resolution) = sizex; allSizey(resolution) = sizey; 
     end
    
     
     % filtering images at multiple resolutions
     SUM1mapFind = applyfilter_MultiResolution(ImageMultiResolution, filters, halfFilterSizes, nOrient,locationShiftLimit,orientShiftLimit,isLocalNormalize,isSeparateLocalNormalize,localHalfx,localHalfy,thresholdFactor);
     mapName = ['feature/SUMMAXmap-' 'image' num2str(iImg)];   
     save(mapName, 'ImageMultiResolution', 'SUM1mapFind','allSizex', 'allSizey');                
     
     disp(['filtering time: ' num2str(toc) ' seconds']);
   
end



%% Prepare variables for EM

    %%%%%%%%%%%%%%%%%%%% show intialization
% % % cluster=cell(numCluster,1);
% % % for i=1:size(dataWeightAll,1)
% % %     ind=find(dataWeightAll(i,:)==max(dataWeightAll(i,:)));
% % %     ind=ind(1);
% % %     cluster{ind}=[cluster{ind},i];
% % % end
% % %     
    %%%%% generating html
% % %     fid = fopen(['result0.html'], 'wt'); 
% % % 
% % %     for c=1:numCluster
% % %        for i=1:size(cluster{c},2)
% % %            id=cluster{c}(i);
% % %            fprintf(fid, '%s\n', ['<IMG SRC="' 'positiveImages/' files(id).name '" height=70 ' 'width=70>']); 
% % %        end
% % %        fprintf(fid, '%s\n', ['<br>']);
% % %        
% % %        fprintf(fid, '%s\n', ['<hr>']);
% % %     end
% % % 
% % %     fprintf(fid, '%s\n', ['<br>']);
% % %     fprintf(fid, '%s\n', ['<hr>']);
% % %     fclose(fid);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


cropedImage=single(zeros(sx, sy));
SUM1mapLearn = cell(numImage, numFilter); 
for (img = 1:numImage) 
    for (iFilter = 1:numFilter)
       SUM1mapLearn{img, iFilter} = single(zeros(sx, sy)); 
    end
end


% intitialization for the first run of M-step
ind=originalResolution;   % original resolution
MrotAll = zeros(numImage, numCluster);    % 0 level
MflipAll = zeros(numImage, numCluster);   % no flip
MindAll = ones(numImage, numCluster) * ind;  % original resolution
MFxAll = ones(numImage, numCluster) * floor(allSizex(ind)/2);  % center x
MFyAll = ones(numImage, numCluster)* floor(allSizey(ind)/2);   % center y
MAX2scoreAll = rand(numImage, numCluster);   % random 
rHat = cell(numFilter,1);
%% EM iteration
for (it = 1 : numEMIteration)
   
   copy_MAX2scoreAll=MAX2scoreAll;
   
   cluster=cell(numCluster,1);
   for (c = 1:numCluster)
      
      disp(['M-step of iteration ' num2str(it) 'for cluster' num2str(c)]);
      savingFolder=['results/iteration' num2str(it) '/cluster' num2str(c) '/'];
      if ~exist(savingFolder)
         mkdir(savingFolder);
      end
      
      
      %% learning for cluster c
      tic      
      t=0;
      for iImg = 1:numImage
        if (copy_MAX2scoreAll(iImg, c) == max(copy_MAX2scoreAll(iImg, :)))
           mapName = ['feature/SUMMAXmap-' 'image' num2str(iImg)];   
           load(mapName);    % load ImageMultiResolution, SUM1mapFind, allSizex, allSizey for iImg
           Mrot = MrotAll(iImg, c); Mflip = MflipAll(iImg, c); 
           Mind = MindAll(iImg, c); MFx = MFxAll(iImg, c); MFy = MFyAll(iImg, c); 
           t = t + 1; 
           for (iF = 1: 2)
              for (orient = 1 : nOrient)
                    orient1 = orient - Mrot; 
                    if (orient1 > nOrient)
                           orient1 = orient1 - nOrient; 
                    end
                    if (orient1 <= 0)
                           orient1 = orient1 + nOrient; 
                    end                    
                    Ccopy(SUM1mapLearn{t, orient+(iF-1)*nOrient}, SUM1mapFind{Mind, orient1+(iF-1)*nOrient}, MFx, MFy, floor(sx/2), floor(sy/2), sx, sy, allSizex(Mind), allSizey(Mind), Mrot*pi/nOrient);  
               end
           end
        
           if (Mflip > 0)
              SUM1mapLearnTmp = SUM1mapLearn(t, :); 
              for (iF = 1: 2)  
                   for (iOrient = 1 : nOrient)
                     if (iOrient-1>0)
                       o = nOrient - (iOrient-1) + 1; 
                     else
                       o = iOrient;   
                     end
                     SUM1mapLearn{t, o+(iF-1)*nOrient} = fliplr(SUM1mapLearnTmp{1, iOrient+(iF-1)*nOrient}); 
                   end
              end
           end
           
           %%% cropped the training images for cluster c
           Ccopy(cropedImage, single(ImageMultiResolution{Mind}), MFx, MFy, floor(sx/2), floor(sy/2), sx, sy, allSizex(Mind), allSizey(Mind), Mrot*pi/nOrient);
           if(Mflip > 0) 
             cropedImage=fliplr(cropedImage);
           end
           gLow = min(cropedImage(:));
           gHigh = max(cropedImage(:));
           img_tem = (cropedImage-gLow)/(gHigh-gLow);
           imwrite(img_tem,[ 'working/' 'cropped-iteration-' num2str(it) '-cluster-' num2str(c) '-training-' num2str(iImg,'%04d') '.png']);
           %%%%%
                      
           cluster{c}=[cluster{c},iImg]; % collect the id for each cluster

           
        end
      end
      
      for iFilter = 1:numFilter
          rHat{iFilter}=zeros(sx, sy);
      end
    
      for iImg = 1:t
        for iFilter = 1:numFilter
          rHat{iFilter}= rHat{iFilter}+double(SUM1mapLearn{iImg,iFilter});
        end
      end
      
      for iFilter = 1:numFilter
          rHat{iFilter}=rHat{iFilter}/t;
      end
      
      isSaved=1;
      isComputelogZ=1;
      [lambdaF,currSample,logZ]=FRAMElearnGPU(nIter,filters,rHat,numFilter, sx, sy, halfFilterSizes,nTileRow,nTileCol,epsilon,L,lambdaLearningRate,numSample, isSaved,savingFolder,isComputelogZ);
      disp(['learning time for cluster ' num2str(c) ' takes ' num2str(toc) ' seconds']);
      
       %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% ratate template (lambda)
      disp(['rotating template of cluster' num2str(c)]);
      alllambda=cell((2*rotateShiftLimit+1)*(1+flipOrNot), numFilter);
      for (flip = 0 : flipOrNot)
    
        if (flip > 0)
            flip_lambdaF=cell(1,numFilter);
        
            for (iF = 1: 2)  
              for iOrient = 1:nOrient
                if (iOrient-1>0)
                    o = nOrient - (iOrient-1) + 1; 
                else
                    o = iOrient;   
                end
                flip_lambdaF{o+(iF-1)*nOrient} = fliplr(lambdaF{iOrient+(iF-1)*nOrient});
              end
            end
        
           lambdaF=flip_lambdaF;
        end
        for (rot = -rotateShiftLimit : rotateShiftLimit)
            r = rot+rotateShiftLimit+1  + (rotateShiftLimit*2+1)*flip;  
            angle=rot*180/nOrient;    
           for (iF = 1: 2)   
             for iOrient = 1:nOrient
                 o=iOrient-rot;
                if(o>nOrient)
                 o=o-nOrient;
                end
                if(o<=0)
                 o=o+nOrient;
                end
                alllambda{r, o+(iF-1)*nOrient}= imrotate(lambdaF{iOrient+(iF-1)*nOrient},-angle,'bilinear','loose');
             end
           end
        end
                
     end
      %%%%%%%%%%%%%%%% end of  ratating the template
      
      %%%%%% cluster c scores every image by detetion process allowing template local shift 
      disp(['E-step of iteration ' num2str(it) 'for cluster' num2str(c)]);
      
      for (img = 1 : numImage)
        tic  
        mapName = ['feature/SUMMAXmap-' 'image' num2str(img)];   
        load(mapName);    
                 
        MMAX2 = -1e10; 
        
        for (flip = 0 : flipOrNot)
          for (rot = -rotateShiftLimit : rotateShiftLimit)
            
             r = rot+rotateShiftLimit+1 + (rotateShiftLimit*2+1)*flip;  
            [allFx, allFy,MAX2score,SUM2] = FRAME_SUM2_logZ(numResolution, allSizex, allSizey, numFilter, alllambda(r, :), logZ, SUM1mapFind,halfTemplatex,halfTemplatey,halfFilterSizes(1));  
            
            [maxOverResolution, ind] = max(MAX2score);   % most likely resolution
            if (MMAX2 < maxOverResolution) 
               MMAX2 = maxOverResolution; Mrot = rot; Mflip = flip; 
               Mind = ind; MFx = allFx(ind); MFy = allFy(ind); 
               MSUM2 = SUM2;
            end
          end
        end
        
        %save(['working/iteration-' num2str(it) '-image' num2str(img) 'cluster-' num2str(c) '.mat'],'allFx', 'allFy','MAX2score','MSUM2','Mrot','Mflip','MFx','MFy','Mind');  
        %%%%% draw bounding box
                
        scaledImg=ImageMultiResolution{Mind};
        rect_Vertices=[-halfTemplatex -halfTemplatey; -halfTemplatex halfTemplatey; halfTemplatex halfTemplatey; halfTemplatex -halfTemplatey; -halfTemplatex -halfTemplatey];
        top_left_Vertices=[-halfTemplatex-0.1*halfTemplatex -halfTemplatey; -halfTemplatex-0.1*halfTemplatex -halfTemplatey+0.1*halfTemplatey; -halfTemplatex -halfTemplatey+0.1*halfTemplatey];
        combined_Vertices=[rect_Vertices; top_left_Vertices];
        if Mflip==1 
           combined_Vertices(:,2)=-1*combined_Vertices(:,2); 
        end
       
        theta = Mrot*pi/nOrient;
        R=[cos(theta) -sin(theta);sin(theta) cos(theta)];    %clockwise rotate theta
       
        new_rect_Vertices=combined_Vertices*R+ones(size(combined_Vertices,1),1)*[MFx, MFy];
        fig=figure; imshow(scaledImg,[]); hold on;
        plot(new_rect_Vertices(:,2),new_rect_Vertices(:,1),'LineWidth',1.5);
        saveas(fig,[ 'working/iteration-' num2str(it) 'cluster-' num2str(c) '-training-' num2str(img,'%04d') ],'png'); %name is a string
        %%%%%%
        
        
        MAX2scoreAll(img, c) = MMAX2; 
        MrotAll(img, c) = Mrot; MflipAll(img, c) = Mflip; 
        MindAll(img, c) = Mind; MFxAll(img, c) = MFx; MFyAll(img, c) = MFy;   
        
        disp(['detection time for image ' num2str(img) ' takes ' num2str(toc) ' seconds']);
      end
      close all;
      
              
    end
 
    
    
    
    
    %%%%% generating html
    fid = fopen(['result' num2str(it) '.html'], 'wt'); 

    for c=1:numCluster
       for i=1:size(cluster{c},2)
           id=cluster{c}(i);
           img_name=[ 'working/' 'cropped-iteration-' num2str(it) '-cluster-' num2str(c) '-training-' num2str(id,'%04d') '.png'];
           fprintf(fid, '%s\n', ['<IMG SRC="' img_name '" height=70 ' 'width=70>']); 
       end
       fprintf(fid, '%s\n', ['<br>']);
       fprintf(fid, '%s\n', ['<hr>']);
    end

    fprintf(fid, '%s\n', ['<br>']);
    fprintf(fid, '%s\n', ['<hr>']);
    fclose(fid);
                               
   
end %it




