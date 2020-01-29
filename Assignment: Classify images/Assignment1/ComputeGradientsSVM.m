function [grad_Wsvm, grad_bsvm] = ComputeGradientsSVM(X, Y, Psvm, W, lambda)
    [k,d] = size(W);
    grad_Wsvm = zeros(k,d);
    N = size(X,2);
    grad_bsvm = zeros(k,1);

    for i = 1:N
       label = 0;
       for j = 1:k
            if Y(j,i)==1
                  label = j;
            end
       end
       p = Psvm(:,i);
       for j=1:k
            if (p(j,1)-p(label,1)+1)>0
                grad_Wsvm(label,:) = grad_Wsvm(label,:) - X(:,i)';
                grad_bsvm(label,:) = grad_bsvm(label,:) - 1;
            end
       end
    end

    grad_bsvm = grad_bsvm/N;
    grad_Wsvm = grad_Wsvm/N + 2*lambda*W;
    
end
