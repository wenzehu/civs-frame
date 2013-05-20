% main function for the FRAME model for general images
%% Step 0: prepare filter and training images
f0 = MakeFilter(0.5,8);
f1 = MakeFilter(0.9,8);
f = [f0 f1];
filters = f;
numFilter = length(f);
numSample = 100;

img = imresize(imread('./image_0023.jpg'),0.3);
img = rgb2gray(img);
img = ceil(im2double(img)*8);
%% Step 1: compute training sample averages
rHat = cell(numFilter,1);
for iFilter = 1:numFilter
    rHat{iFilter} = zeros(size(img,1),size(img,2));
end

for iFilter = 1:numFilter
    Y = filter2(filters{iFilter},img);
    rHat{iFilter}= rHat{iFilter}+abs(Y);
end
%% Step 2: optimize the exponential model
% our model parameter
lambdaF = cell(numFilter,1);
gradientF = cell(numFilter,1);
for iFilter = 1:numFilter
    lambdaF{iFilter} = zeros(size(img));%rand(size(img))/1e5;
    gradientF{iFilter}= zeros(size(img));
end
currSample = randi(8,size(img));
% store the <F, I> of current sample
rSample = cell(numFilter,1);
for iFilter = 1:numFilter
    rSample{iFilter}=filter2(filters{iFilter},currSample);
end

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
                % compute conditional energy
                for iFilter = 1:numFilter
                        halfSize = (size(filters{iFilter},1)-1)/2;
                        xVec = max(1,cx-halfSize):min(size(img,1),cx+halfSize);
                        yVec = max(1,cy-halfSize):min(size(img,2),cy+halfSize);
                        [x, y]=meshgrid(xVec,yVec);
                        lambdaFIndex = reshape(sub2ind(size(img),x,y),numel(x),1);
                        [x, y]=meshgrid(cx-xVec+halfSize+1,cy-yVec+halfSize+1);
                        filterPosIndex =  reshape(sub2ind(size(filters{iFilter}),x,y),numel(x),1); 
                    for iVal = 1:8
                        newRSample = filters{iFilter}(filterPosIndex).*(iVal-v0) + rSample{iFilter}(lambdaFIndex);
                        localEnergy(iVal) = localEnergy(iVal)+sum(sum(abs(newRSample).*lambdaF{iFilter}(lambdaFIndex)));
                    end
                end
                localEnergy = localEnergy-min(localEnergy);
                expState = exp(localEnergy);
                pState = expState/norm(expState);
                cumPState = cumsum(pState);
                uRand = rand(1);
                v1 = find(cumPState>=uRand);
                
                currSample(cx,cy)=v1(1);
                
                % update rSample
                iVal = v1(1);
                for iFilter = 1:numFilter
                    halfSize = (size(filters{iFilter},1)-1)/2;
                    xVec = max(1,cx-halfSize):min(size(img,1),cx+halfSize);
                    yVec = max(1,cy-halfSize):min(size(img,2),cy+halfSize);
                    [x, y]=meshgrid(xVec,yVec);
                    lambdaFIndex = reshape(sub2ind(size(img),x,y),numel(x),1);
                    [x, y]=meshgrid(cx-xVec+halfSize+1,cy-yVec+halfSize+1);
                    filterPosIndex =  reshape(sub2ind(size(filters{iFilter}),x,y),numel(x),1);
                    rSample{iFilter}(lambdaFIndex) = filters{iFilter}(filterPosIndex).*(iVal-v0) + rSample{iFilter}(lambdaFIndex);
                end
            end
            cx 
        end
        disp( [ 'sample ' num2str(iSample) ' of ' num2str(numSample)]);
        % update the model mean
        for iFilter = 1:numFilter
            rModel{iFilter} = rModel{iFilter} + abs(rSample{iFilter})/numSample;
        end
    end
    
    % compute gradient and do graidnet ascent
    step_width = step_width *0.93;
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
% Step 3: synthesis images.