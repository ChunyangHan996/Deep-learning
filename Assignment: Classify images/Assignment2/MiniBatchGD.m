function [Wstar, bstar] = MiniBatchGD(X, Y, X1,Y1,GDparams, W, b, lambda, rho,decay_rate)

    N = size(X,2);
    n_batch = GDparams(1,1);
    eta = GDparams(1,2);
    n_epochs = GDparams(1,3);
    
    cost = ComputeCost(X, Y, W, b, lambda);
    trainingloss=[cost];
    vcost = ComputeCost(X1, Y1, W, b, lambda);
    validationloss=[vcost];
    
    vw1 = zeros(size(W{1}));
    vw2 = zeros(size(W{2}));
    vb1 = zeros(size(b{1}));
    vb2 = zeros(size(b{2}));
  
    for i = 1:n_epochs
        for j=1:N/n_batch
            j_start = (j-1)*n_batch + 1;
            j_end = j*n_batch;
            Xbatch = X(:, j_start:j_end);
            Ybatch = Y(:, j_start:j_end);
            P = EvaluateClassifier(Xbatch,W,b);
            [grad_W, grad_b] = ComputeGradients(Xbatch, Ybatch, P, W,b, lambda);
             
            vw1 = rho* vw1 + eta*grad_W{1};
            vw2 = rho* vw2 + eta*grad_W{2};
            vb1 = rho* vb1 + eta*grad_b{1};
            vb2 = rho* vb2 + eta*grad_b{2};
            
            W{1} = W{1} - vw1;
            W{2} = W{2} - vw2;
            b{1} = b{1} - vb1;
            b{2} = b{2} - vb2;

        end
 
        temp = ComputeCost(X, Y, W, b, lambda);
        temp1= ComputeCost(X1,Y1,W,b,lambda);
        if(temp > 3*cost)
            break;
        end
        trainingloss = [trainingloss;temp];% cost
        validationloss=[validationloss;temp1];
 
        eta = eta* decay_rate;  
    end
       
    sizes = size(validationloss,1);
    plot(1:sizes,validationloss,'r');
    hold on
    plot(1:sizes,trainingloss,'g');
    hold off
    legend('validation loss','training loss');
    title('Momentum with rho=0');
    xlabel('epoch');
    ylabel('loss');
    axis auto;

    Wstar = W;
    bstar = b;
      
end