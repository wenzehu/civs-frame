img = imread('./testZebra.jpg');
model = load('./10by10zebra/model.mat');

img = im2double(rgb2gray(imresize(img,128/1800)));
%img = gpuArray(im2double(img));
img = img-mean(img(:));
img = img/std(img(:));

[sx sy]=size(model.lambdaF{1});
[SX SY]=size(img);

%{
padImg = randn(SX+2*sx, SY+2*sy);
padImg(sx+1:sx+SX, sy+1:sy+SY)=img;
img = padImg;
%}


gaussianFilter = ones(sx,sy)/2;

accS2map = zeros(size(img));
for iFilter = 1:numel(model.filters)
    S1map = abs(imfilter(img,model.filters{iFilter}));
    S2map = imfilter(S1map,model.lambdaF{iFilter},100);
    accS2map=S2map+accS2map;
end
accS2map = accS2map + imfilter(S1map.^2,gaussianFilter);
accS2map = gather(accS2map);
accS2map1=accS2map;
%accS2map1=accS2map(sx+1:sx+SX,sy+1:sy+SY);
imshow(accS2map1,[])
[val ind]=max(accS2map1(:));
[x y]=ind2sub([SX,SY],ind);
imshow(img,[]);
line([y-64 y-64],[x-64 x+64])
line([y+64 y+64],[x-64 x+64])
line([y-64 y+64],[x-64 x-64])
line([y-64 y+64],[x+64 x+64])
