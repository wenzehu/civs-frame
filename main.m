% main function for the FRAME model for general images
%% Step 0: prepare filter and training images
f0 = MakeFilter(0.7,8);
f1 = MakeFilter(2,8);
f = [f0 f1];
filters = cell(length(f)*2,1);
for i = 1:length(f)
    filters{i*2-1}=real(f{i});
    filters{i*2}=imag(f{i});
end
numFilter = length(f)*2;
step_width = 1e-3;

img = imresize(imread('./image_0023.jpg'),0.5);
img = rgb2gray(img);
img = ceil(im2double(img)*8);
%% Step 1: compute training sample averages
rHat = cell(numFilter,1);
for iFilter = 1:numFilter
    rHat{iFilter} = zeros(size(img,1),size(img,2));
end

for iFilter = 1:numFilter
    Y = filter2(filters{iFilter},img);
    rHat{iFilter}= rHat{iFilter}+Y;
end
%% Step 2: optimize the exponential model
% our model parameter
lambdaF = cell(numFilter,1);
for iFilter = 1:numFilter
    lambdaF{iFilter} = rand(size(img));
end
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
                lambdaI(cx,cy)=lambdaI(cx,cy)+lambdaF{iFilter}(lambdaFIndex)*filters{iFilter}(filterPosIndex);
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
                lambdaI(cx,cy)=lambdaI(cx,cy)+lambdaF{iFilter}(lambdaFIndex)*filters{iFilter}(filterPosIndex);
            end
        end
        % disp([ num2str(cx) ' of ' num2str(size(img,1))] );
    end
    
    % computing the histogram for each pixel
    pixelHist = zeros(numel(lambdaI),8);
    for val = 1:8
        pixelHist(:,val) = reshape(lambdaI*val,numel(lambdaI),1);
    end
    pixelHist= exp(pixelHist);
    zz = 1./sum(pixelHist,2);
    for val = 1:8
        pixelHist(:,val) = pixelHist(:,val).*zz;
    end
    cumHist = cumsum(pixelHist,2);
    numSample=100+iter*10;
    
    % draw samples
    disp('drawing samples');
    for iSample = 1:numSample
        randomNumbers = rand(numel(img),1);
        sampleImg = zeros(numel(img),1);
        for val=1:8
            ind = find(randomNumbers<=cumHist(:,val));
            sampleImg(ind)=val;
            randomNumbers(ind)=Inf;
        end
        sampleImg = reshape(sampleImg,size(img));
        
        % filter sampleImg, so as to compute image statistics
        for iFilter = 1:numFilter
            rMap = filter2(filters{iFilter},sampleImg);
            rModel{iFilter}=rModel{iFilter}+rMap/numSample;
        end
        if mod(iSample,100)==0
            disp(['generated' num2str(iSample) ' of ' num2str(numSample) ' samples']);
        end
    end
    % compute gradient and do graidnet ascent
    step_width = step_width *0.93;
    for iFilter = 1:numFilter
        gradient = rHat{iFilter}-rModel{iFilter};
        lambdaF{iFilter}=lambdaF{iFilter}+ step_width*gradient;
    end
    % save synthesied image
    imwrite(sampleImg/8,[ num2str(iter,'%04d') '.png']);
end
%% Step 3: synthesis images.
