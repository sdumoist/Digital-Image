info = imfinfo('Img4.gif');
len = length(info);
for i = 1 : len
    [Ii, map] = imread('Img4.gif', 'frames', i);
    I(:, i) = im2frame(Ii, map);
end
movie(I);
