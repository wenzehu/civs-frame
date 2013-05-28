function [allFiltered] = applyfilter_MultiResolution(I, filters, halfFilterSize, nOrient,locationShiftLimit,orientShiftLimit,LocalNormOrNot,isSeparate,localHalfx,localHalfy,thresholdFactor)
% filter image by a bank of filters
% I: input images
% allFilter: filter bank
numImage = size(I, 2);    % number of images
numFilter = size(filters, 2);   % number of orientations
allFiltered = cell(numImage, numFilter);  % filtered images
useDoG= mod(numFilter,nOrient);

for iImg = 1:numImage
    S1 = cell(numFilter,1);
    M1 = cell(numFilter,1);
    
    % SUM1
    
    [sx,sy]=size(I{iImg});
    for iFilter = 1:numFilter
        Y = filter2(filters{iFilter},I{iImg});
        S1{iFilter} = abs(single(Y));
        M1{iFilter} = zeros(sx,sy,'single');
    end
    if LocalNormOrNot
       h1 = halfFilterSize(1);
       if isSeparate
         [S1(1:nOrient)]= ...
         LocalNormalize(S1(1:nOrient),[],h1,round(0.6*h1),round(0.6*h1),thresholdFactor);
         [S1(nOrient+1: 2*nOrient)]= ...
         LocalNormalize(S1(nOrient+1: 2*nOrient),[],h1,round(0.6*h1),round(0.6*h1),thresholdFactor);

         %{
         h2 = halfFilterSize(2*nOrient+1); 
         [S1(2*nOrient+1:3*nOrient)]= ...
         LocalNormalize(S1(2*nOrient+1:3*nOrient),[],h2,round(0.6*h2),round(0.6*h2),thresholdFactor);
         [S1(3*nOrient+1: 4*nOrient)]= ...
         LocalNormalize(S1(3*nOrient+1: 4*nOrient),[],h2,round(0.6*h2),round(0.6*h2),thresholdFactor);
         %}
       else
         [S1(1:nOrient),S1(nOrient+1: 2*nOrient)]= ...
         LocalNormalize(S1(1:nOrient),S1(nOrient+1: 2*nOrient),h1,round(0.6*h1),round(0.6*h1),thresholdFactor);
         %{
         h2 = halfFilterSize(2*nOrient+1); 
         [S1(2*nOrient+1:3*nOrient),S1(3*nOrient+1: 4*nOrient)]= ...
         LocalNormalize(S1(2*nOrient+1:3*nOrient),S1(3*nOrient+1: 4*nOrient),h2,round(0.6*h2),round(0.6*h2),thresholdFactor);
         %}
        end
       if useDoG
       h = halfFilterSize(end);
       S1(end)=LocalNormalize(S1(end),[],h,round(0.6*h),round(0.6*h),thresholdFactor);
       end
    end
        
    
    %MAX1
    CgetMAX1(1,sx,sy,nOrient,locationShiftLimit,orientShiftLimit,1,S1(1:nOrient),M1(1:nOrient));
    CgetMAX1(1,sx,sy,nOrient,locationShiftLimit,orientShiftLimit,1,S1(nOrient+1: 2*nOrient),M1(nOrient+1: 2*nOrient));
    %{
    CgetMAX1(1,sx,sy,nOrient,locationShiftLimit,orientShiftLimit,1,S1(2*nOrient+1:3*nOrient),M1(2*nOrient+1:3*nOrient));
    CgetMAX1(1,sx,sy,nOrient,locationShiftLimit,orientShiftLimit,1,S1(3*nOrient+1: 4*nOrient),M1(3*nOrient+1: 4*nOrient));
    %}
    if useDoG 
          M1{end}=S1{end};
    end
    allFiltered(iImg,:)=M1;
end
