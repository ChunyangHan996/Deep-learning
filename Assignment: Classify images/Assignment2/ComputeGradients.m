function [grad_W, grad_b] = ComputeGradients(X, Y, P, W, b, lambda)
% DENOTE d as the dimensionality of each image, N as the number of images 
%        K as the number of label kinds
% INPUT     - X:                d*N
%           - Y:                K*N
%           - P:                K*N
%           - W:                K*d
%           - lambda:           1*1
% OUTPUT    - grad_W:           K*d
%           - grad_b:           K*1
    grad_W = cell(numel(W), 1);
    grad_b = cell(numel(b), 1);
    grad_W{1} = zeros(size(W{1}));
    grad_W{2} = zeros(size(W{2}));
    grad_b{1} = zeros(size(b{1}));
    grad_b{2} = zeros(size(b{2}));
    N = size(X,2);
    for i = 1:N
        x = X(:,i);
        y = Y(:,i);
        p = P(:,i);
        g = ( -y'/(y'*p))*(diag(p) - p*p');
        s = W{1}*x + b{1};
        h = W{1}*x + b{1};
        h = max(0,h);
        ind = zeros(size(s));
        for k = 1:size(s,1)
            if(s(k,:)>0)
                ind(k,:) = 1;
            end
        end
        grad_W{2} = grad_W{2} + g'*h';
        grad_b{2} = grad_b{2} + g';
        g = g*W{2}*diag(ind); 
        
        grad_W{1} = grad_W{1} + g'*x';
        grad_b{1} = grad_b{1} + g';
    end
    for i = 1:2
        grad_W{i} = grad_W{i}/N + 2*lambda*W{i};
        grad_b{i} = grad_b{i}/N;
    end 
end


