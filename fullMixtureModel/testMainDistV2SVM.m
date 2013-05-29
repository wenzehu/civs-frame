dataPath = 'C:/Users/wzhu/Documents/Image';
resultPath ='C:/Users/wzhu/Dropbox/civs-frame/animalExpV2LMSD';
modelPath = 'C:/Users/wzhu/Dropbox/civs-frame/animalExpV2LMSD/models';
%% setup the cluster machine
parallel.defaultClusterProfile('test01');
c = parcluster('test01');
tmp_list= ListAttachedFiles;
matlabpool(c);
matlabpool('addattachedfiles',tmp_list);
%% read in testing images
load(fullfile(resultPath,'split.mat'));
nCate= max(testingLabels);
nTest = length(testingImages);
testImageCell=cell(nTest,1);
for iTest = 1:nTest
    folderName = cateNames{testingLabels(iTest)};
    testImageCell{iTest} = imread(fullfile(dataPath,folderName,testingImages(iTest).name));
end

nTrain = length(trainingImages);
trainImageCell = cell(nTrain,1);
for iTrain = 1:nTrain
    folderName = cateNames{trainingLabels(iTrain)};
    trainImageCell{iTrain}=imread(fullfile(dataPath,folderName,trainingImages(iTrain).name));
end

%% testing stage
sharedParametersV2
f1 = MakeFilter(0.7,nOrient);
for i=1:nOrient
    f1_r{i} =real(f1{i});
    f1_i{i} =imag(f1{i});
end
filters = [f1_r f1_i];
f2 = MakeFilter(0.35,nOrient);
for i=1:nOrient
    f2_r{i} =real(f2{i});
    f2_i{i} =imag(f2{i});
end
filters = [f1_r f1_i f2_r f2_i];
if useDoG
    f0  = single(dog(8,0));
    filters = [filters f0];
end
numFilter = length(filters);
for iFilter = 1:numFilter
    filters{iFilter}=(single(filters{iFilter}));
end
trainScoreMatrix = zeros(nCate,nTrain);
testScoreMatrix = zeros(nCate,nTest);

for iCate = 1:nCate
    % load testing models
    cate_model_path = fullfile(modelPath,cateNames{iCate});
    cate_model_name= fullfile(cate_model_path,'model_final.mat');
    disp(['trying to find file ' cate_model_name]);
    while ~exist(cate_model_name,'file');
        pause(60);
    end
    pause(10);
    model = load(cate_model_name,'clusters');
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
            for sF= 0:1 % scale of filter
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
            for sF=0:1
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
    % assemble the inputs
    % run the job
	disp('begin computing scores');
    parfor iTest=1:nTest
        testScoreMatrix(iCate,iTest)=testOneImageDistV2(testImageCell{iTest},alllambda,logZ,filters);
    end
    parfor iTrain = 1:nTrain
        trainScoreMatrix(iCate,iTrain)=testOneImageDistV2(trainImageCell{iTrain},alllambda,logZ,filters);
    end
    % run inference on each testing images
    save(fullfile(resultPath,['scoreMat' num2str(iCate) '.mat']),'trainScoreMatrix','testScoreMatrix');
    disp(['finished category ' cateNames{iCate}	' at' datestr(now)]);
end
matlabpool close
save(fullfile(resultPath,['scoreMat_final.mat']),'trainScoreMatrix','testScoreMatrix');
%% SVM training 



load(fullfile(resultPath,['scoreMat_final.mat']),'trainScoreMatrix','testScoreMatrix');
for iCate = 1:nCate
    % load testing models
    cate_model_path = fullfile(modelPath,cateNames{iCate});
    cate_model_name= fullfile(cate_model_path,'model_final.mat');
    model = load(cate_model_name,'clusters');
    logZ= model.clusters.logZ;
    trainScoreMatrix(iCate,:)=trainScoreMatrix(iCate,:)+logZ;
    testScoreMatrix(iCate,:)=testScoreMatrix(iCate,:)+logZ;
end

addpath('./liblinear-1.93/matlab');
libLinearOptions='-s 0 -c 1 -B 1' ; % no bias if -B<0
model = train(trainingLabels',sparse(trainScoreMatrix'),libLinearOptions);
[predicted_label, accuracy]=predict(testingLabels',sparse(testScoreMatrix'),model);

%{
[confMat,AF]=makeConfMat(testingLabels,scoreMatrix);
disp(confMat)
disp(AF)
save(fullfile(resultPath,['confMat' num2str(iCate) '.mat']),'confMat','AF');
%}
