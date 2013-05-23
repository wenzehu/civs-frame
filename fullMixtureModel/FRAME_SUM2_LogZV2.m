function [allFx, allFy,MAX2score,SUM2] = FRAME_SUM2_LogZV2(numResolution, allSizex, allSizey, numFilter, lambdaF, logZ, SUM1mapFind,halfTempSizex,halfTempSizey,Mh)
allFx = zeros(1,numResolution);
allFy = zeros(1,numResolution);
MAX2score = zeros(1,numResolution)-Inf;
for iResolution=1:numResolution
    [sx sy]=size(SUM1mapFind{iResolution,1});
    [lsx lsy]=size(lambdaF{1});
    if lsx>sx || lsy>sy % filter is larger than image
       continue;
    end
    SUM2map = filter2(lambdaF{1},SUM1mapFind{iResolution,1},'valid');
    [fsx fsy]=size(SUM2map);
    for iFilter=2:numFilter
     SUM2map = SUM2map + filter2(lambdaF{iFilter},SUM1mapFind{iResolution,iFilter},'valid');
    end
    SUM2map = SUM2map-logZ;  
    SUM2map = gather(SUM2map);
    [MAX2score(iResolution),ind]=max(SUM2map(:));
    [indX,indY]=ind2sub([fsx fsy],ind);
    allFx(iResolution)=ceil(indX+(sx-fsx)/2);
    allFy(iResolution)=ceil(indY+(sy-fsy)/2);
end% iResolution
end % function
 
