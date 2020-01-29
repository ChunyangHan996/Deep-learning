function [Wstar, bstar] = MiniBatchGD1(X, Y,GDparams, W, b, lambda)
n_batch = GDparams(1,1); %the size of the mini-batches
eta = GDparams(1,2);   % the learning rate
e_epochs = GDparams(1,3); %the number of runs through the whole training set.
N = size(X,2);
J = ComputeCost(X, Y, W, b, lambda);

for i = 1:e_epochs
    for j = 1:N/n_batch
        j_start = (j-1)*n_batch + 1;
        j_end = j*n_batch;
        inds = j_start:j_end;
        Xbatch = X(:,inds);
        Ybatch = Y(:,inds);
        
        P=EvaluateClassifier(Xbatch,W,b);
        [grad_W, grad_b] = ComputeGradients(Xbatch, Ybatch, P, W, lambda);
        W = W - grad_W*eta;
        b = b - grad_b*eta;
    end
    J = [J,ComputeCost(X, Y, W, b, lambda)];
    
end
Wstar = W;
bstar = b;
end



