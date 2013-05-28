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
imageCell=cell(nTest,1);
for iTest = 1:nTest
    folderName = cateNames{testingLabels(iTest)};
    imageCell{iTest} = imread(fullfile(dataPath,folderName,testingImages(iTest).name));
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
scoreMatrix = zeros(nCate,nTest);
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
        scoreMatrix(iCate,iTest)=testOneImageDistV2(imageCell{iTest},alllambda,logZ,filters);
    end
    %{
    inputDataCell = cell(nTest,1);
    for iTest = 1:nTest
           inputDataCell{iTest}={imageCell{iTest},alllambda,logZ,filters};
    end
    nBatch= ceil(nTest/batchSize);
    for iBatch = 1:nBatch
    job1 = createJob(c);
    job1.AttachedFiles = tmp_list;
    createTask(job1,@testOneImageDist,1,inputDataCell(iBatch:iBatch+batchSize-1));
    submit(job1);
    wait(job1);
    partialResults = fetchOutputs(job1);
    partialRsults = cell2mat(partialResults);
    scoreMatrix(iCate,iBatch:iBatch+batchSize-1)=partialResults;
    delete(job1);
    end
    %}
    % run inference on each testing images
    save(fullfile(resultPath,['scoreMat' num2str(iCate) '.mat']),'scoreMatrix');
    disp(['finished category ' cateNames{iCate}	' at' datestr(now)]);
end

[confMat,AF]=makeConfMat(testingLabels,scoreMatrix);
disp(confMat)
disp(AF)
save(fullfile(resultPath,['confMat' num2str(iCate) '.mat']),'confMat','AF');
matlabpool close
