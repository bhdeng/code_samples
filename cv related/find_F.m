function F_p = find_F(x1,x2)
    % compute A
    A = zeros(size(x1,2),9);
    for i=1:size(x1,2)
        x_p = x2(1,i);
        y_p = x2(2,i);
        x = x1(1,i);
        y = x1(2,i);
        A(i,:) = [x_p*x, x_p*y, x_p, y_p*x, y_p*y, y_p, x, y, 1];
    end
    
    % solve f from SVD
    [U,D,V] = svd(A);
    n = size(A,2);
    f = V(:,n);
    % construct F from f
    F = reshape(f,3,3)';
    % solve F_p under constraints
    [Uf,Df,Vf] = svd(F);
    Df(3,3) = 0;
    F_p = Uf*Df*Vf';
end