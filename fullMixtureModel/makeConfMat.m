function [confMat,AF]=makeConfMat(labels,scoreMat);
% asume scoreMat is a marix of (nCate, nImage), where each entry is the logP
[V,idx]=max(scoreMat);
[nCate, nTest]=size(scoreMat);
T = zeros(nCate,nTest);
Y = zeros(nCate,nTest);
for iTest= 1:nTest
    T(labels(iTest),iTest)=1;
    Y(idx(iTest),iTest)=1;
end

[AF, confMat] = confusion(T,Y);
for iRow = 1:size(confMat,1)
    confMat(iRow,:)=confMat(iRow,:)/sum(confMat(iRow,:));
end
AF=1-AF;
%AF = mean(diag(confMat));
%
