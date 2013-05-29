% function to split the data and train models using the training images
dataPath = '~/Documents/civs-frame/Image'
resultPath ='~/Dropbox/civs-frame/animalExpV3'
modelPath = '~/Dropbox/civs-frame/animalExpV3/models';
cachePath = '~/Documents/civs-frame/animalExpV3/cache';
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
load(fullfile(resultPath,'split.mat'));
nTest= length(testingImages);
%% call training procedure
nCate = max(trainingLabels);
for iCate = nCate:-1:1
	cate_trainImgs = trainingImages(find(trainingLabels==iCate));
	cate_model_path = fullfile(modelPath,cateNames{iCate});
	cate_cache_path = fullfile(cachePath,cateNames{iCate});
	if exist(cate_model_path,'dir')
		disp(['Error: the folder ' resultPath 'already exists. Please delete before runing'])
		return;
	end
	em_mixtureFRAMEFuncV3(cate_trainImgs,cate_model_path,cate_cache_path);
end
