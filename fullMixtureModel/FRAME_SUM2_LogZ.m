function [allFx, allFy,MAX2score,SUM2] = FRAME_SUM2_LogZ(numResolution, allSizex, allSizey, numFilter, lambdaF, logZ, SUM1mapFind,halfTempSizex,halfTempSizey,Mh)

allFx = zeros(1,numResolution);
allFy = zeros(1,numResolution);
MAX2score = zeros(1,numResolution);
SUM2=cell(1,numResolution);
for iResolution=1:numResolution
    SUM2{iResolution}=gpuArray(zeros(allSizex(iResolution), allSizey(iResolution),'single'));
    
    for iFilter=1:numFilter
        %% option 3
%         [temp_x,temp_y]=size(lambdaF{iFilter});
%         SUM1map=randn(allSizex(iResolution)+temp_x-1,allSizey(iResolution)+temp_y-1);
%         start_x=round((temp_x-1)/2)+1;
%         start_y=round((temp_y-1)/2)+1;
%         SUM1map(start_x:start_x+allSizex(iResolution)-1, start_y:start_y+allSizey(iResolution)-1)=SUM1mapFind{iResolution,iFilter}-mean(SUM1mapFind{iResolution,iFilter}(:));
%         SUM2{iResolution}=SUM2{iResolution}+filter2(lambdaF{iFilter},SUM1map,'valid');
        %% option 2
 %       SUM1map=SUM1mapFind{iResolution,iFilter}-mean(SUM1mapFind{iResolution,iFilter}(:));
 %       SUM2{iResolution}=SUM2{iResolution}+filter2(lambdaF{iFilter},SUM1map);
        %%  option 1
        %SUM2{iResolution}=SUM2{iResolution}+filter2(lambdaF{iFilter},SUM1mapFind{iResolution,iFilter});
        %%  option 0
        SUM2{iResolution} =SUM2{iResolution} + imfilter(SUM1mapFind{iResolution,iFilter},lambdaF{iFilter},100);
    end
    SUM2{iResolution}=SUM2{iResolution}-logZ;
    
    SUM2{iResolution}=gather(SUM2{iResolution});
    % inhibite invalid template positions
    SUM2{iResolution}([1:halfTempSizex+Mh,end-halfTempSizex-Mh+1:end],:)=-1e10;
    SUM2{iResolution}(:,[1:halfTempSizey+Mh,end-halfTempSizey-Mh+1:end])=-1e10;
    [MAX2score(iResolution),ind]=max(SUM2{iResolution}(:));
    [allFx(iResolution),  allFy(iResolution)] = OffSet2XY(ind,allSizex(iResolution));
end% iResolution
end % function
 
function [x,y]=OffSet2XY(ind, nRows)
   x=mod((ind-1),nRows)+1;
   y=floor((ind-1)/nRows)+1;

end
