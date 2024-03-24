function gaussian(input_img,sigma)
   input = imread(input_img);  
   subplot(1,2,1);
   imshow(input);
   input = double(input);
   [y,x,dimension] = size(input);
   output = uint8(zeros(y,x,dimension));
  
  
   n = floor(6*sigma-1)/2*2+1;%窗口大小
   cen = floor((n+1)/2);%图像中心
   g = zeros(n);%n*n的矩阵
   sum = 0.0;
   %一维高斯函数
   for i = 1:n
       g(i) = exp(-((i-cen).^2)/(2*sigma^2))/(sqrt(2*pi*sigma));
       sum = sum + g(i);
   end
   %归一化
   for i = 1:n
       g(i) = g(i)/sum;
   end
   
   d = (n-1)/2;%卷积核中心到边界的距离

   R = input(:,:,1);
   G = input(:,:,2);
   B = input(:,:,3);
   R2 = uint8(zeros(y,x));
   G2 = uint8(zeros(y,x));
   B2 = uint8(zeros(y,x));
   %y方向上的滤波
   for i = 1:x
       for j = 1-d:y-d%将卷积中心作为原点
           R_sum = 0;
           G_sum = 0;
           B_sum = 0;
           temp = 0;
           for m = -d:d%与中心的距离
               if((m+j)>=1&&(m+j)<=y)
                   R_sum = R_sum + (R(m+j,i)*g(d+m+1));
                   G_sum = G_sum + (G(m+j,i)*g(d+m+1));
                   B_sum = B_sum + (B(m+j,i)*g(d+m+1));
                   temp = temp + g(d+m+1);
               end
           end
           R2(m+j,i) = R_sum/temp;
           G2(m+j,i) = G_sum/temp;
           B2(m+j,i) = B_sum/temp;
       end
    end
    %x方向上的滤波
    for j = 1:y
       for i = 1-d:x-d%将卷积中心作为原点
           R_sum = 0;
           G_sum = 0;
           B_sum = 0;
           temp = 0;
           for m = -d:d%与中心的距离
               if((m+i)>=1&&(m+i)<=x)
                   R_sum = R_sum + (R2(j,m+i)*g(d+m+1));
                   G_sum = G_sum + (G2(j,m+i)*g(d+m+1));
                   B_sum = B_sum + (B2(j,m+i)*g(d+m+1));
                   temp = temp + g(d+m+1);
               end
           end
           R(j,m+i) = R_sum/temp;
           G(j,m+i) = G_sum/temp;
           B(j,m+i) = B_sum/temp;
       end
    end

    output(:,:,1) = R;
    output(:,:,2) = G;
    output(:,:,3) = B;

    subplot(1,2,2);
    imshow(output);
    imwrite(output,'test3_1.png');
end






