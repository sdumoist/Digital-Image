function changeshape(input_img)
    input = imread(input_img);
    [height,width,dim] = size(input);
    for i = 1:width
        for j = 1:height
            a = (i-0.5*width)/(0.5*width);
            b = (j-0.5*height)/(0.5*height);
            r  = sqrt(a^2 + b^2);
            seita = (1-r)^2;
            if r >= 1
               x = a;
               y = b;
            else
               x = cos(seita)*a - sin(seita)*b;
               y = sin(seita)*a + cos(seita)*b;
            end
               x1 = uint16((x + 1)*0.5*width);
               y1 = uint16((y + 1)*0.5*height);
            output(j,i,:) = input(y1,x1,:);
        end

    end
  imshow(output);
  imwrite(output,'test2_2.png');
end