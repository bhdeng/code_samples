function crad = blob_detect()
    im = imread('3.bmp');
    I = rgb2gray(im);
    I = im2double(I);

    % smooth the image with gaussian
    G = fspecial('gaussian', [3,3], 1);
    I = conv2(I, G, 'same');

    % convolution with laplacian of gaussian kernel
    I_filter = [];
    scale = 0.5:1.25:18;
    for iter=1:length(scale)
        sigma = scale(iter);
        window_size = 2*ceil(3*sigma)+1;
        LoG = sigma^2*fspecial('log', window_size, sigma);
        I_filter = cat(3, I_filter, conv2(I, LoG, 'same'));
    end
    % find the characteristic scale
    width = size(I_filter, 1);
    height = size(I_filter, 2);
    M = zeros(width, height);
    radii = zeros(width, height);
    for i=1:width
        for j=1:height
            [m, index] = max(I_filter(i,j,:));
            M(i,j) = m;
            radii(i,j) = sqrt(2)*scale(index);
        end
    end
    % non-max suppression
    T = 0.18;
    crad = [];
    cx = [];
    cy = [];
    for i=2:width-1
        for j = 2:height-1
            if M(i,j) > T && M(i,j) >= max(max(M(i-1:i+1, j-1:j+1)))
                cx = [cx; i];
                cy = [cy; j];
                crad = [crad; radii(i,j)];
            end
        end
    end
    show_all_circles(I, cy, cx, crad);
end