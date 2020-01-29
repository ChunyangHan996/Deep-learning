function [Wstar, bstar] = MiniBatchGDSVM(X, Y, X1,Y1,GDparams, W, b, lambda)
    N = size(X,2);% return # of A's column
    n_batch = GDparams(1,1);
    eta = GDparams(1,2);
    n_epochs = GDparams(1,3);
    cost = ComputeCostSVM(X, Y, W, b, lambda);
    trainingloss=[cost];
    validationloss=[cost];
    
    for i = 1:n_epochs
        for j=1:N/n_batch
            j_start = (j-1)*n_batch + 1;
            j_end = j*n_batch;
            inds = j_start:j_end;
            Xbatch = X(:, j_start:j_end);
            Ybatch = Y(:, j_start:j_end);
            P = EvaluateClassifierSVM(Xbatch,W,b);
            [grad_W, grad_b] = ComputeGradientsSVM(Xbatch, Ybatch, P, W, lambda);
            W = W - eta * grad_W;
            b = b - eta * grad_b;
        end
        temp = ComputeCostSVM(X, Y, W, b, lambda);
        temp1=ComputeCostSVM(X1,Y1,W,b,lambda);
        trainingloss = [trainingloss;temp];% cost
        validationloss=[validationloss;temp1];
    end
      %for plot  
    plot(1:n_epochs+1, trainingloss,'g-',1:n_epochs+1,validationloss,'r-');
    title('n\_epoch=40,n\_batch=100,eta=.01,and lambda=0')
    xlabel('epoch');
    ylabel('loss');
    legend('SVMTrainingloss','SVMValidationloss')
    hold on;
    axis auto;
    
    Wstar = W;
    bstar = b;
end