% main function for the FRAME model for general images
nOrient = 8;
%% Step 0: prepare filter and training images
mex CgetMAX1.c;
f0 = MakeFilter(0.5,nOrient);
h0 = (size(f0{1},1)-1)/2;
f1 = MakeFilter(1,nOrient);
h1 = (size(f1{1},1)-1)/2;
f2 = dog(10,0);
h0 = (size(f2,1)-1)/2;
f = [f0 f1 f2];
filters = f;
numFilter = length(f);
numSample = 1;

inPath = './positiveImage';
imgCell = cell(0);
files = dir([inPath '/*.jpg']);
for iImg = 1:length(files)
    img = imread(fullfile(inPath,files(iImg).name));
    img = imresize(img,[100,100]);
    imgCell{iImg}=ceil(im2double(rgb2gray(img))*8);
end
%% Step 1: compute training sample averages
sx = size(img,1);
sy = size(img,2);
img = rgb2gray(img);
rHat = cell(numFilter,1);
for iFilter = 1:numFilter
    rHat{iFilter}=zeros(sx,sy);
end
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
    CgetMAX1(1,sx,sy,nOrient,3,1,1,S1(1:nOrient),M1(1:nOrient));
    CgetMAX1(1,sx,sy,nOrient,3,1,1,S1(nOrient+1:2*nOrient),M1(nOrient+1:2*nOrient));
    M1{2*nOrient+1}=S1{2*nOrient+1};
    for iFilter = 1:numFilter
        rHat{iFilter}= rHat{iFilter}+double(M1{iFilter});
    end
end
for iFilter = 1:numFilter
    rHat{iFilter}=rHat{iFilter}/length(files);
end
disp('finished filtering');

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
            filterIndex{cx,cy,iFilter}=filterPosIndex;
            lambdaIndex{cx,cy,iFilter}=lambdaFIndex;
        end
    end
end


step_width = 5e-2;
for iter = 1:100
    disp( [ 'iteration: ' num2str(iter)]);
    % rModel, to store the model mean
    rModel = cell(numFilter,1);
    for iFilter = 1:numFilter
        rModel{iFilter} = zeros(size(img,1),size(img,2));
    end
    for iSample = 1:numSample
        for cx  = 1:size(img,1)
            for cy = 1:size(img,2)
                localEnergy = zeros(8,1);
                v0 = currSample(cx,cy);
                
                
                for iFilter = 1:numFilter
                    filterPosIndex = filterIndex{cx,cy,iFilter};
                    lambdaFIndex = lambdaIndex{cx,cy,iFilter};
                    for iVal = 1:8
                        newRSample = filters{iFilter}(filterPosIndex).*(iVal-v0) + rSample{iFilter}(lambdaFIndex);
                        localEnergy(iVal) = localEnergy(iVal)+sum(sum(abs(newRSample).*lambdaF{iFilter}(lambdaFIndex)));
                    end
                end
                
                localEnergy = localEnergy-min(localEnergy);
                expState = exp(localEnergy);
                pState = expState/sum(expState);
                
                
                cumPState = cumsum(pState);
                uRand = rand(1);
                v1 = find(cumPState>=uRand);
                currSample(cx,cy)=v1(1);
                
                % update rSample
                iVal = v1(1);
                for iFilter = 1:numFilter
                    filterPosIndex = filterIndex{cx,cy,iFilter};
                    lambdaFIndex = lambdaIndex{cx,cy,iFilter};
                    rSample{iFilter}(lambdaFIndex) = filters{iFilter}(filterPosIndex).*(iVal-v0) + rSample{iFilter}(lambdaFIndex);
                end
            end
        end
        disp( [ 'sample ' num2str(iSample) ' of ' num2str(numSample)]);
        % update the model mean
        for iFilter = 1:numFilter
            rModel{iFilter} = rModel{iFilter} + abs(rSample{iFilter})/numSample;
        end
    end
    
    % compute gradient and do graidnet ascent
    step_width = step_width *0.9;
    lambdaNorm = 0;
    gradientNorm =0;
    for iFilter = 1:numFilter
        gradientF{iFilter} = rHat{iFilter}-rModel{iFilter};
        gradientNorm = gradientNorm + norm(gradientF{iFilter});
    end
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

