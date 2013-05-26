function colorImg= gray2rgb(img)
colorImg= zeros([size(img) 3],'uint8');
for iCh = 1:3
    colorImg(:,:,iCh)=img;
end
