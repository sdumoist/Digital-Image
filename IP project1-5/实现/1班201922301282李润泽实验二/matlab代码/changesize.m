function changesize(input_img,y,x)
  input = imread(input_img);
  %[I,map] = imread(input_img);
  [width,height,dim] = size(input);
  w = round(width*x);
  h = round(height*y);
   for i = 1:w
      for j = 1:h
          a = floor((i-1)/x);
          b = floor((j-1)/y);
          a1 = (i-1)/x;
          b1 = (j-1)/y;
          if a == 0 || b == 0 || a == width-1 || b == height-1
              output(1,j,:) = input(1,b+1,:);
              output(i,1,:) = input(a+1,1,:);
          else
              a = a+1;
              b = b+1;
              output(i,j,:) = input(a,b,:)*(a-a1)*(b-b1)+...
                              input(a,b+1,:)*(a-a1)*(b1-b+1)+...
                              input(a+1,b,:)*(a1-a+1)*(b-b1)+...
                              input(a+1,b+1,:)*(a1-a+1)*(b1-b+1);
          end
      end
   end
  imshow(output);
  imwrite(output,'test2_1.png');
 end

