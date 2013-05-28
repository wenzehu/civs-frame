% feature
sx =71;
sy =71;
halfTemplatex=floor(sx/2);
halfTemplatey=floor(sy/2);
padding_x=round(halfTemplatex/3);
padding_y=round(halfTemplatey/3);
nOrient = 16;
locationShiftLimit=0;
orientShiftLimit=0;
localHalfx=20;
localHalfy=20;
isLocalNormalize=true; %
isSeparateLocalNormalize=false;
thresholdFactor=0.01;
useDoG=true;
% tepmlate learning
nTileRow = 15; %nTileRow \times nTileCol defines the number of paralle chains
nTileCol = 15;
lambdaLearningRate = 0.01;
nIter = 5; % the number of iterations for learning lambda
epsilon = 0.01; % step size for the leapfrog
L = 10; % leaps for the leapfrog
numSample = 3; % how many HMC calls for each learning iteration
isSaved=1;
isComputelogZ=1;
isWarmStart = 1; % Learn template using its last iteration as starting point

% unkown resolution, location, orientation, flip
numResolution=3;
scaleStepSize=0.2;
flipOrNot=1;            % template flip Or not
rotateShiftLimit = 3;   % template rotation  from -rotateShiftLimit to rotateShiftLimit, eg. -2:2 if rotateShiftLimit=2

% mixture model
numEMIteration=5;
numCluster=1;
