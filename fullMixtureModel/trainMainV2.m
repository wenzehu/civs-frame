% function to split the data and train models using the training images
mex CgetMAX1.c;
mex Ccopy.c; % copy around detect location
dataPath = '~/Documents/civs-frame/Image'
resultPath ='~/Dropbox/civs-frame/animalExpV2LMSD'
modelPath = '~/Dropbox/civs-frame/animalExpV2LMSD/models';
cachePath = '~/Documents/civs-frame/animalExpV2LMSD/cache';
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
%matlabpool open 4;
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
