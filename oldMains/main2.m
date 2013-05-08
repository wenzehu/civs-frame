% main function for the FRAME model for general images
%% Step 0: prepare filter and training images
f0 = MakeFilter(0.5,8);
f1 = MakeFilter(1.5,8);
f = [f0 f1];
filters = f;
f = [f0 f1];
filters = cell(length(f)*2,1);
for i = 1:length(f)
    filters{i*2-1}=real(f{i});
    filters{i*2}=imag(f{i});
end
numFilter = length(f)*2;
step_width = 1e-2;

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
    rHat{iFilter}= rHat{iFilter}+Y.^2;
end
%% Step 2: optimize the exponential model
% our model parameter
lambdaF = cell(numFilter,1);
gradientF = cell(numFilter,1);
for iFilter = 1:numFilter
    lambdaF{iFilter} = zeros(size(img));%rand(size(img))/1e5;
    gradientF{iFilter}= zeros(size(img));
end
currSample = zeros(size(img));%randi(8,size(img));
for iter = 1:100
    disp( [ 'iteration: ' num2str(iter)]);
    % rModel, to store the model mean
    rModel = cell(numFilter,1);
    for iFilter = 1:numFilter
        rModel{iFilter} = zeros(size(img,1),size(img,2));
    end
    
    disp('computing lambdaI')
    lambdaI = zeros(size(img)); % lambda over pixels
    for cx  = 1:size(img,1)
        for cy = 1:size(img,2)
            iFilter = 1;
            halfSize = (size(filters{iFilter},1)-1)/2;
            xVec = max(1,cx-halfSize):min(size(img,1),cx+halfSize);
            yVec = max(1,cy-halfSize):min(size(img,2),cy+halfSize);
            [x, y]=meshgrid(xVec,yVec);
            lambdaFIndex = reshape(sub2ind(size(img),x,y),1,numel(x));
            [x, y]=meshgrid(cx-xVec+halfSize+1,cy-yVec+halfSize+1);
            filterPosIndex =  reshape(sub2ind(size(filters{iFilter}),x,y),numel(x),1);
            for iFilter = 1:16
                lambdaI(cx,cy)=lambdaI(cx,cy)+lambdaF{iFilter}(lambdaFIndex)*(filters{iFilter}(filterPosIndex).^2);
            end
            iFilter = 17;
            halfSize = (size(filters{iFilter},1)-1)/2;
            xVec = max(1,cx-halfSize):min(size(img,1),cx+halfSize);
            yVec = max(1,cy-halfSize):min(size(img,2),cy+halfSize);
            [x, y]=meshgrid(xVec,yVec);
            lambdaFIndex = reshape(sub2ind(size(img),x,y),1,numel(x));
            [x, y]=meshgrid(cx-xVec+halfSize+1,cy-yVec+halfSize+1);
            filterPosIndex =  reshape(sub2ind(size(filters{iFilter}),x,y),numel(x),1);
            for iFilter = 17:numFilter
                lambdaI(cx,cy)=lambdaI(cx,cy)+lambdaF{iFilter}(lambdaFIndex)*(filters{iFilter}(filterPosIndex).^2);
            end
        end
        % disp([ num2str(cx) ' of ' num2str(size(img,1))] );
    end
    
    
    numSample = 100 %+ 10*iter;
    % compute the  energy for current sample
    energy = 0;
    for iFilter = 1:numFilter
        rMap = filter2(filters{iFilter},currSample);
        energy  = energy + sum(sum(lambdaF{iFilter}.*(rMap.^2)));
    end
    for iSample= 1:numSample
        disp(['draw samples' num2str(iSample) ])
        for cx = 1:size(img,1)
            for cy = 1:size(img,2)
                v0 = currSample(cx,cy);
		iVal = 1:8;
		pState = lambdaI(cx,cy)*(iVal.^2-v0^2) + energy;
                expState = exp(pState);
                normExpState = expState/sum(expState);
                cumPState = cumsum(normExpState);
                uRand = rand(1); % uniform random variable
                v1 = find(cumPState>=uRand);
                v1 = v1(1);
                energy = energy + lambdaI(cx,cy)*(v1^2-v0^2);
                currSample(cx,cy)=v1;
            end
        end
        
        % after a sweep, update the model mean;
        for iFilter = 1:numFilter
            rMap = filter2(filters{iFilter},currSample);
            rModel{iFilter} = rModel{iFilter}+rMap.^2/numSample;
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
    for iFilter = 1:numFilter
	lambdaF{iFilter}=lambdaF{iFilter}+ step_width*gradientF{iFilter}/gradientNorm;
   	lambdaNorm = lambdaNorm + norm(lambdaF{iFilter});
    end
    for iFilter = 1:numFilter
	lambdaF{iFilter} = lambdaF{iFilter}/lambdaNorm;
    end
    % save synthesied image
    imwrite(currSample/8,[ num2str(iter,'%04d') '.png']);
end
% Step 3: synthesis images.
