function [grad_W, grad_b] = ComputeGradients(X, Y, P, W, lambda)
    % Initialization
    [k,d] = size(W);
    grad_W = zeros(k,d);
    grad_b = zeros(k,1);
    N = size(X,2); 
    for i = 1:N
        x = X(:,i);
        y = Y(:,i);
        p = P(:,i);
        
        g = -y'/(y'*p);
        
        g = g*(diag(p) - p*p');
        
        grad_W = grad_W + g'*x';
        grad_b = grad_b + g';
    end    
  
    grad_W = grad_W/N + 2*lambda*W; 
    grad_b = grad_b/N;
end