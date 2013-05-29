function score= testOneImageDistV3(img,allLambdaF,logZ,filters)
	sharedParametersV2;
	originalResolution = round(numResolution/2);    %3; % original resolution is the one at which the imresize factor = 1, see 11th line beneath this line
	allSizex = zeros(1, numResolution);
	allSizey = zeros(1, numResolution);
	ImageMultiResolution = cell(1, numResolution);

    %img = imread(imgName);
    img = imresize(img,[sx+padding_x, sy+padding_y]);
    if size(img,3)==1
       img = gray2rgb(img);
    end
    %colorImg = padarray(img,[padding_x padding_y],'replicate');
    
    img = im2single(rgb2gray(img));
    %img = padarray(img,[padding_x padding_y],'replicate');
    img = img-mean(img(:));
    img = img/std(img(:));
    % create image pyramid
    scaleFactors= zeros(numResolution,1);
    for(resolution=1:numResolution)
        resizeFactor = 1.0 + (resolution - originalResolution)*scaleStepSize;
        scaleFactors(resolution)=resizeFactor;
        ImageMultiResolution{resolution} = (imresize(img, resizeFactor, 'nearest'));  % images at multiple resolutions
        [sizex, sizey] = size(ImageMultiResolution{resolution});
        allSizex(resolution) = sizex; allSizey(resolution) = sizey;
    end
    halfFilterSizes=zeros(size(filters));
    for iFilter = 1:numel(filters)
        halfFilterSizes(iFilter)=(size(filters{iFilter},1)-1)/2;
    end
    numFilter= numel(filters);
    SUM1mapFind = applyfilter_MultiResolution(ImageMultiResolution, filters, halfFilterSizes, nOrient,locationShiftLimit,orientShiftLimit,isLocalNormalize,isSeparateLocalNormalize,localHalfx,localHalfy,thresholdFactor);    
    % filtering images at multiple resolutions
    MMAX2=-Inf;
    for r = 1:size(allLambdaF,1)
        [~, ~,MAX2score] = FRAME_SUM2_LogZV3(numResolution, [], [], numFilter, allLambdaF(r, :), logZ,ImageMultiResolution, SUM1mapFind,round(sx/2),round(sy/2),[]);
        MMAX2 = max(MMAX2,max(MAX2score));
    end
    score = MMAX2;
end
