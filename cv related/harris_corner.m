function harris_corner(img_name)
    im = imread(img_name);
    I = rgb2gray(im);
    I = im2double(I);

    % filter the image with Gaussian to reduce noise
    K = fspecial('gaussian', 3, 1);
    I = conv2(I, K, 'same');
    % use sobel kernels to approximate Derivative of Gaussian
    Gx = [-1 0 1; -2 0 2; -1 0 1];
    Gy = [1 2 1; 0 0 0; -1 -2 -1];

    % convolution to get derivatives at each pixel location
    Ix = conv2(I, Gx, 'same');
    Iy = conv2(I, Gy, 'same');
    Ix2 = Ix .* Ix;
    Iy2 = Iy .* Iy;
    Ixy = Ix .* Iy;

    % parameters
    window_size = 7;
    sigma = 2;
    k = 0.06;
    T = 0.1;
    
    % convolution by a gaussian window kernel
    G = fspecial('gaussian', window_size, sigma);
    Sx2 = conv2(Ix2, G, 'same');
    Sy2 = conv2(Iy2, G, 'same');
    Sxy = conv2(Ixy, G, 'same');

    width = size(Sx2,1);
    height = size(Sx2,2);
    R = zeros(width, height);

    for i = 2:width-1
        for j = 2:height-1
            % construct matrix M for each pixel
            M = [Sx2(i,j), Sxy(i,j); Sxy(i,j), Sy2(i,j)];
            R(i,j) = det(M) - k*(trace(M))^2;
        end
    end
    % plot ellipse in a portion
    imshow(im);
    hold on;
%     for i=100:120
%         for j=20:40
%             M = [Sx2(i,j), Sxy(i,j); Sxy(i,j), Sy2(i,j)];
%             [V,D] = eig(M);
%             w = D(1,1); %^(-0.5);
%             h = D(2,2); %^(-0.5);
%             theta = atan2(V(2,2), V(1,2));
%             plot_ellipse(w, h, j, i, theta*90/pi, 'r');
%         end
%     end
    % non-max suppression
    corners = zeros(width, height);
    % get rid of the edge
    for i=3:width-2
        for j = 3:height-2
            if R(i,j) > T && R(i,j) >= max(max(R(i-1:i+1, j-1:j+1)))
                corners(i,j) = 1;
            end
        end
    end
    [corner_y, corner_x] = find(corners==1);
    plot(corner_x, corner_y, 'rx');
end
