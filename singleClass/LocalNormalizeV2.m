function S1LocalNorm=LocalNormalize(S1,dummy,h,localHalfx,localHalfy,thresholdFactor)

% h is the half of filter size
% localHalfx and localHalfy are window size

norient = length(S1);
[Sx,Sy] = size(S1{1});

mask = ones(Sx,Sy);
mask([1:h end-h+1:end],:) = 0;
mask(:,[1:h end-h+1:end]) = 0;


S1LocalMean = cell(1,norient);
localAveFilter = ones(localHalfx*2+1,localHalfy*2+1,'single')/single(localHalfx*2+1)/single(localHalfy*2+1);
for o = 1:norient
    S1LocalMean{o} = filter2(localAveFilter,S1{o}.^2,'same');
end

S1LocalMeanOverOrientation = zeros(Sx,Sy);
for o = 1:norient
    S1LocalMeanOverOrientation = S1LocalMeanOverOrientation + 1/norient * S1LocalMean{o};
end
S1LocalMeanOverOrientation=sqrt(S1LocalMeanOverOrientation);

maxAverage = max(S1LocalMeanOverOrientation(:));
S1LocalMeanOverOrientation = max(S1LocalMeanOverOrientation,maxAverage*thresholdFactor);

S1LocalNorm = cell(1,norient);
for o = 1:norient
    S1LocalNorm{o} = S1{o} ./ S1LocalMeanOverOrientation .* mask;
end

Upperbound = 6; % upperbound for Gabor response
% sigmoid transformation
Ctransform(1, norient, S1LocalNorm,Sx,Sy, Upperbound);
