[confMat,AF]=makeConfMat(labels,scoreMat);
% asume scoreMat is a marix of (nCate, nImage), where each entry is the logP
[~,idx]=max(scoreMat);
[nCate, nTest]=size(scoreMat);
T = zeros(nCate,nTest);
Y = zeros(nCate,nTest);
for iTest= 1:nTest
    T(labels(iTest),iTest)=1;
    Y(idx(iTest),iTest)=1;
end

confMat = confusion(T,Y);
AF = mean(diag(confMat));
%
