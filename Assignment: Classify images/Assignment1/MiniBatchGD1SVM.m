function [Wstar, bstar] = MiniBatchGD1SVM(X, Y,GDparams, W, b, lambda)
    N = size(X,2);% return # of A's column
    n_batch = GDparams(1,1);
    eta = GDparams(1,2);
    n_epochs = GDparams(1,3);
    cost = ComputeCostSVM(X, Y, W, b, lambda);
    trainingloss=[cost];
    for i = 1:n_epochs
        for j=1:N/n_batch
            j_start = (j-1)*n_batch + 1;
            j_end = j*n_batch;
            inds = j_start:j_end;
            Xbatch = X(:, inds);
            Ybatch = Y(:, inds);
            P = EvaluateClassifierSVM(Xbatch,W,b);
            [grad_W, grad_b] = ComputeGradientsSVM(Xbatch, Ybatch, P, W, lambda);
            W = W - eta * grad_W;
            b = b - eta * grad_b;
        end
        temp = ComputeCostSVM(X, Y, W, b, lambda);
        
        trainingloss = [trainingloss;temp];% cost
        
    end
    
    Wstar = W;
    bstar = b;
end