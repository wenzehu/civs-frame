function score= testOneImage(imgName,allLambdaF,logZ,filters)
	sharedParameters;
	originalResolution = round(numResolution/2);    %3; % original resolution is the one at which the imresize factor = 1, see 11th line beneath this line
	allSizex = zeros(1, numResolution);
	allSizey = zeros(1, numResolution);
	ImageMultiResolution = cell(1, numResolution);

    img = imread(imgName);
    img = imresize(img,[sx, sy]);
    if size(img,3)==1
       img = gray2rgb(img);
    end
    %colorImg = padarray(img,[padding_x padding_y],'replicate');
    
    img = im2single(rgb2gray(img));
    img = padarray(img,[padding_x padding_y],'replicate');
    img = img-mean(img(:));
    img = img/std(img(:));
    % create image pyramid
    scaleFactors= zeros(numResolution,1);
    for(resolution=1:numResolution)
        resizeFactor = 1.0 + (resolution - originalResolution)*scaleStepSize;
        scaleFactors(resolution)=resizeFactor;
        ImageMultiResolution{resolution} = gpuArray(imresize(img, resizeFactor, 'nearest'));  % images at multiple resolutions
        [sizex, sizey] = size(ImageMultiResolution{resolution});
        allSizex(resolution) = sizex; allSizey(resolution) = sizey;
    end
    
    % filtering images at multiple resolutions
    numFilter = length(filters);
    S1 = cell(numResolution,numFilter);
    for iResolution = 1:numResolution
    for iFilter = 1:numFilter
    	h = (size(filters{iFilter},1)-1)/2;
    	map = filter2(filters{iFilter},ImageMultiResolution{iResolution});
    	map([1:h end-h:end],:)=0;
    	map(:,[1:h end-h:end])=0;
    	S1{iResolution,iFilter}=map;
    end
    end
    MMAX2= -Inf;
    for r = 1:size(allLambdaF,1)
        [~, ~,MAX2score] = FRAME_SUM2_LogZV2(numResolution, [], [], numFilter, allLambdaF(r, :), logZ, S1,round(sx/2),round(sy/2),[]);
        MMAX2 = max(MMAX2,max(MAX2score));
    end
    score = MMAX2;
end
