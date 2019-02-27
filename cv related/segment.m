function img=segment(img,k)
    img = im2double(img);
    width = size(img,1);
    height = size(img,2);

    % initialize data structures
    e = 1e-10;
    compare = zeros(width*height,k);
    n_mean = zeros(k,3);
    % reshape the img
    img = reshape(img,width*height,3);

    % initialization
    i_pixels = rand(k,3);
    
    % while
    while 1
        % calculate distance to the k means
        for i=1:k
            centroid = i_pixels(i,:);
            compare(:,i) = sqrt(sum(bsxfun(@minus, img, centroid).^2,2));
        end
        % assign cluster
        [~,I] = min(compare,[],2);
        % recalculate means for k clusters
        for i=1:k
            n_mean(i,:) = mean(img(I==i,:));
        end
        % check condition
        if abs(max(i_pixels-n_mean)) < e
            break
        end
        i_pixels = n_mean;
    end

    % segmentation result
    colors = [1,0,0;
              0,1,0;
              0,0,1];
    for i=1:k
        result = I == i;
        img(result,:) = repmat(i_pixels(i,:), sum(result(:)), 1);
    end
    img = reshape(img, width, height, 3);
end