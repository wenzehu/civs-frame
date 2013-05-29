function [allFx, allFy,MAX2score,SUM2] = FRAME_SUM2_LogZV3(numResolution, allSizex, allSizey, numFilter, lambdaF,logZ,multiResImage, SUM1mapFind,halfTempSizex,halfTempSizey,Mh)
allFx = zeros(1,numResolution);
allFy = zeros(1,numResolution);
MAX2score = zeros(1,numResolution)-Inf;
dx = round(halfTempSizex*.2);
dy = round(halfTempSizey*.2);

for iResolution=1:numResolution
    [sx sy]=size(SUM1mapFind{iResolution,1});
    [lsx lsy]=size(lambdaF{1});
    if lsx>sx || lsy>sy % filter is larger than image
       continue;
    end
    gaussianFilter = ones(size(lambdaF{1}),'single');
    SUM2map = filter2(gaussianFilter,multiResImage{iResolution}.^2,'valid');
    SUM2map = -SUM2map/2;
    [fsx fsy]=size(SUM2map);
    for iFilter=1:numFilter
     SUM2map = SUM2map + filter2(lambdaF{iFilter},SUM1mapFind{iResolution,iFilter},'valid');
    end
    SUM2map = SUM2map-logZ;  
    SUM2map = gather(SUM2map);
%{
    start_x = max(1,round(fsx/2-dx));
    end_x = min(fsx,round(fsx/2+dx));
    start_y = max(1,round(fsy/2-dy));
    end_y = min(fsy, round(fsy/2+dy));
    SUM2map = SUM2map(start_x:end_x,start_y:end_y);
%}
    [fsx fsy]=size(SUM2map);
    [MAX2score(iResolution),ind]=max(SUM2map(:));
    [indX,indY]=ind2sub([fsx fsy],ind);
    allFx(iResolution)=ceil(indX+(sx-fsx)/2);
    allFy(iResolution)=ceil(indY+(sy-fsy)/2);
end% iResolution
end % function
 
