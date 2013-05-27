% function to split the data and train models using the training images
mex CgetMAX1.c;
mex Ccopy.c; % copy around detect location
disp('warning, start from iCate=6')
dataPath = '~/Documents/civs-frame/Image'
resultPath ='~/Dropbox/civs-frame/animalExpV2'
modelPath = '~/Dropbox/civs-frame/animalExpV2/models';
cachePath = '~/Documents/civs-frame/animalExpV2/cache';
if exist(cachePath,'dir')
   rmdir(cachePath,'s')
end
mkdir(cachePath);
if exist(modelPath,'dir')
	disp(['Model path already exist: ' modelPath])
	%return;
end
mkdir(resultPath);
mkdir(modelPath);
matlabpool open 8;
%% split the data into half for training and half for testing
folders = dir(dataPath);
numCate  = length(folders);
cateNames = cell(0);
trainingImages = struct('path',[],'name',[]);
testingImages = struct('path',[],'name',[]); 
trainingLabels = [];
testingLabels = [];
iTrain = 1; iTest =1;iCate = 1;
for iFolder = 1:numCate
	if folders(iFolder).isdir ==0
		continue;
	end
	folderName = folders(iFolder).name;
	if strcmp(folderName,'.') || strcmp(folderName,'..') || strcmp(folderName,'Natural')
		continue;
	end
	curr_folder = fullfile(dataPath,folderName);
	files = dir(fullfile(curr_folder,'*.jpg'));
	nFiles= length(files);
	splitPoint = round(nFiles/2);
	idx = randperm(nFiles);
	for iImg = 1:splitPoint
		trainingImages(iTrain).path = curr_folder;
		trainingImages(iTrain).name = files(idx(iImg)).name;
		trainingLabels(iTrain)=iCate;
		iTrain = iTrain + 1;
	end
	for iImg = splitPoint+1:nFiles
		testingImages(iTest).path = curr_folder;
		testingImages(iTest).name = files(idx(iImg)).name;
		testingLabels(iTest)=iCate;
		iTest = iTest +1;
	end
	cateNames{iCate}=folderName;
	iCate = iCate + 1;
end
nTrain=iTrain-1;
nTest = iTest-1;
save(fullfile(resultPath,'split.mat'),'trainingImages','testingImages','trainingLabels','testingLabels','cateNames');
%load(fullfile(resultPath,'split.mat'));
nTest= length(testingImages);
%% call training procedure
nCate = max(trainingLabels);
for iCate = 1:nCate
	cate_trainImgs = trainingImages(find(trainingLabels==iCate));
	cate_model_path = fullfile(modelPath,cateNames{iCate});
	cate_cache_path = fullfile(cachePath,cateNames{iCate});
	em_mixtureFRAMEFuncV2(cate_trainImgs,cate_model_path,cate_cache_path);
end

%% testing stage
sharedParameters
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
	f0  = single(dog(8,0));
	filters = [filters f0];
end
numFilter = length(filters);
for iFilter = 1:numFilter
	filters{iFilter}=(single(filters{iFilter}));
end

scoreMatrix = zeros(nCate,nTest);
for iCate = 1:nCate
	% load testing models
	cate_model_path = fullfile(modelPath,cateNames{iCate});
	model = load(fullfile(cate_model_path,'model_final.mat'),'clusters');
	lambdaF = model.clusters.lambdaF;
	logZ= model.clusters.logZ;
	numFilter = numel(lambdaF);
	for iFilter = 1:numFilter
		lambdaF{iFilter}=(lambdaF{iFilter});
	end
	% transform the learned template
    disp(['rotating template of category' num2str(iCate)]);
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
	% run inference on each testing images
	parfor iTest = 1:nTest
		disp(['To image ' num2str(iTest) ' of '  num2str(nTest)])
		imgName= fullfile(trainingImages(iTest).path,trainingImages(iTest).name);
		scoreMatrix(iCate,iTest)=testOneImageV2(imgName,alllambda,logZ,filters);
	end
	save(fullfile(resultPath,['scoreMat' num2str(iCate) '.mat']),'scoreMatrix');
end
