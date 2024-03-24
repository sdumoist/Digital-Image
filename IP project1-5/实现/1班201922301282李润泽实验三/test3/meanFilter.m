function meanFilter(input_img,w)
   input = imread(input_img);
   subplot(1,2,1);
   imshow(input);
   [y,x,dimension]=size(input);
   Z = (2*w+1)*(2*w+1);
   output = uint8(zeros(y,x,dimension));

   R = input(:,:,1);
   G = input(:,:,2);
   B = input(:,:,3);
   S_R = zeros(y,x);
   S_G = zeros(y,x);
   S_B = zeros(y,x);

   for i = 1:y
       if(i==1)
           S_R(1,1) = R(1,1);
           S_G(1,1) = G(1,1);
           S_B(1,1) = B(1,1);
       else
           S_R(i,1) = sum(R(1:i,1));
           S_G(i,1) = sum(G(1:i,1));
           S_B(i,1) = sum(B(1:i,1));
       end
   end

   for i = 1:y
      for j = 2:x
          S_R(i,j) = S_R(i,j-1) + sum(R(1:i,j));
          S_G(i,j) = S_G(i,j-1) + sum(G(1:i,j));
          S_B(i,j) = S_B(i,j-1) + sum(B(1:i,j));
      end
   end

   for i = 1:w+1
      for j = 1:x
         output(i,j,1) = S_R(i,j)/(i*j);
         output(i,j,2) = S_G(i,j)/(i*j);
         output(i,j,3) = S_B(i,j)/(i*j);
      end
   end

   for i = y-w:y
      for j = 1:x
        output(i,j,1) = S_R(i,j)/(i*j);
        output(i,j,2) = S_G(i,j)/(i*j);
        output(i,j,3) = S_B(i,j)/(i*j);
      end
   end

   for j = 1:w+1
      for i = 1:y
        output(i,j,1) = S_R(i,j)/(i*j);
        output(i,j,2) = S_G(i,j)/(i*j);
        output(i,j,3) = S_B(i,j)/(i*j);
      end
   end

   for j = x-w:x
      for i = 1:y
        output(i,j,1) = S_R(i,j)/(i*j);
        output(i,j,2) = S_G(i,j)/(i*j);
        output(i,j,3) = S_B(i,j)/(i*j);
      end
   end

   for i = w+2:y-w
      for j = w+2:x-w
        output(i,j,1) = (1/Z)*(S_R(i+w,j+w)+S_R(i-w-1,j-w-1)-S_R(i+w,j-w-1)-S_R(i-w-1,j+w));
        output(i,j,2) = (1/Z)*(S_G(i+w,j+w)+S_G(i-w-1,j-w-1)-S_G(i+w,j-w-1)-S_G(i-w-1,j+w));
        output(i,j,3) = (1/Z)*(S_B(i+w,j+w)+S_B(i-w-1,j-w-1)-S_B(i+w,j-w-1)-S_B(i-w-1,j+w));
      end
   end

   subplot(1,2,2);
   imshow(output);
   imwrite(output,'test3_2.jpg');

end
