function F = find_nF(x1,x2)
    % get the first transformation
    cen1 = mean(x1,2);
    diff1 = x1(1:2,:) - repmat(cen1(1:2,:), 1, size(x1,2));
    rms1 = mean(sqrt(sum(diff1.^2)));
    T1 = [sqrt(2)/rms1,0,-cen1(1)*sqrt(2)/rms1;...
        0,sqrt(2)/rms1,-cen1(2)*sqrt(2)/rms1;0,0,1];
    
    % get the second transformation
    cen2 = mean(x2,2);
    diff2 = x2(1:2,:) - repmat(cen2(1:2,:), 1, size(x1,2));
    rms2 = mean(sqrt(sum(diff2.^2)));
    T2 = [sqrt(2)/rms2,0,-cen2(1)*sqrt(2)/rms2;...
        0,sqrt(2)/rms2,-cen2(2)*sqrt(2)/rms2;0,0,1];
    
    q1 = T1*x1;
    q2 = T2*x2;
    F = find_F(q1,q2);
    F = T2'*F*T1;
end