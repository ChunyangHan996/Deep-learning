clc;clear;

[trainX,trainY,trainlabels]=LoadBatch('data_batch_1.mat');
[ValidaX,ValidaY,Validalabels]=LoadBatch('data_batch_2.mat');
[testX,testY,testlabels]=LoadBatch('test_batch.mat');

meanX = mean(trainX,2);
trainX = trainX - repmat(meanX, [1,size(trainX,2)]);
ValidaX = ValidaX - repmat(meanX, [1,size(ValidaX,2)]);
testX = testX - repmat(meanX, [1,size(testX,2)]);

d = size(trainX,1);
K = size (trainY,1);
rng(400);
m=50;
std = 0.001;
lambda = 0;
[W, b] = init_para(m, d, K, std);

% 1. Compare Analytic & Numerical Gradient  
%   h=1e-6;
%   batch_size = 100;  
%   CompareGradient(trainX, trainY, W, b, lambda, h,batch_size);
  
% 2. momentum term
n_batch = 100;
eta = 0.02215;
n_epochs = 20;% n_epochs = 10;
GDparams = [n_batch, eta, n_epochs];

%lambda = 1.93e-7;
%rho = 0.9;
 rho = 0;
decay_rate=0.8;
[Wstar,bstar] = MiniBatchGD(trainX, trainY,ValidaX,ValidaY, GDparams, W, b, lambda, rho,decay_rate);
% k = ComputeAccuracy(testX, testlabels,  Wstar, bstar);

% % randomly search proper eta and lambda 
% emin = 0.01 ;
% emax = 0.3;
% lambdamin = 1e-7;
% lambdamax = 0.1;
% ug = [];
% for i =1:50                                                                                                 
%     disp('~~~~~~~~~~~~~~~~~')
%     disp (i)
%   %3. coarse search
% %     e = emin + (emax - emin)* rand(1,1);
% %     eta = 10^e;
% %     e= lambdamin + (lambdamax-lambdamin)*rand(1,1);
% %     lambda = 10^e; 
%   %4. fine  search
% %     e = -1.65 - 0.1* rand(1,1);
% %     eta = 10^e;
% %     e= -6 - 1*rand(1,1);
% %     lambda = 10^e; 
% %     GDparama = [n_batch, eta,n_epochs];
% %     
% %     [Wstar,bstar] = MiniBatchGD(trainX, trainY,ValidaX,ValidaY,GDparams, W, b, lambda, rho,decay_rate);
% %     k = ComputeAccuracy(ValidaX, Validalabels,  Wstar, bstar);
% %     ug = [ug;eta, lambda,k];
% end 
% disp (ug);


%%
function [dw1,dw2,db1,db2]= CompareGradient( X, Y, W, b, lambda, h,batch_size)
     T=3;
     P=EvaluateClassifier(X(:,1:batch_size),W,b);
     w={W{1}(:,1:batch_size);W{2}};
     [ga_W, ga_b] = ComputeGradients(X(1:batch_size, 1:3), Y(:,1:3), P(:,1:3), w, b, lambda);
     [gn_b, gn_W] = ComputeGradsNumSlow(X(1:batch_size, 1:3), Y(:,1:3), w, b, lambda,h);
     
    
    dw1=norm(ga_W{1}-gn_W{1})/max(eps,norm(ga_W{1}) + norm(gn_W{1}));
    dw2=norm(ga_W{2}-gn_W{2})/max(eps,norm(ga_W{2}) + norm(gn_W{2}));
    db1=norm(ga_b{1}-gn_b{1})/max(eps,norm(ga_b{1}) + norm(gn_b{1}));
    db2=norm(ga_b{2}-gn_b{2})/max(eps,norm(ga_b{2}) + norm(gn_b{2}));
    

    disp(dw1);
    disp(dw2);
    disp(db1);
    disp(db2);
end
%%
function accuracy = ComputeAccuracy(X, y, W, b)
N = size(X,2);
accuracy = EvaluateClassifier(X,W,b);
[C,I] = max(accuracy);
sum = 0;
for i = 1:N
    if(I(1,i)==y(1,i))
        sum = sum+1;
    end
end
accuracy = sum/N;
end
%%
function J = ComputeCost(X, Y, W, b, lambda)
   second = lambda * ((norm(W{1},2))^2 + (norm(W{2},2))^2);
   sums = 0;
   N = size(X,2);
   for i = 1:N
       sums = sums + l_cross(X(:,i),Y(:,i),W,b);
   end
   sums = sums/N;
   J = sums + second;
end

function L = l_cross(X,Y,W,b)
    p = EvaluateClassifier(X,W,b);
    L = -log(Y'*p);
end
%%
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
%%
function P = EvaluateClassifier(X, W, b)
% DENOTE d as the dimensionality of each image, N as the number of images 
%        K as the number of label kinds
% INPUT     - X:        d*N
%           - W:        K*d
%           - b:        K*1
% OUTPUT    - P:        K*N
W1 = W{1}; 
b1 = b{1}; % b1:m*1
b1 = repmat(b1, 1, size(X, 2));
h = W1*X + b1; % W1: m*d
h = max(0, h); % h: m*N
W2 = W{2}; 
b2 = b{2}; % b2:K*1
b2 = repmat(b2, 1, size(h, 2));
s = W2*h + b2; % W2: K*m   s: K*N
denorm = repmat(sum(exp(s), 1), size(W2, 1), 1);
P = exp(s)./denorm;
end
%%
function [W, b] = init_para(m, d, K, std)

W1 = std*randn(m, d);
W2 = std*randn(K, m);
b1 = zeros(m, 1);
b2 = zeros(K, 1);
W = {W1, W2};
b = {b1, b2};

end
%%
function [X, Y, y] = LoadBatch(filen)
    addpath ./Datasets/cifar-10-batches-mat/;
    A = load(filen);
    I = reshape(A.data', 32, 32, 3, 10000);
  
    X = double(reshape(I,3072, 10000))/255;% now data are between 0~1
    y = A.labels'+ones(1,size(A.labels',1));%1~10 instead of  0~9
    Y = zeros(10,size(y,2));

    for i = 1:size(X,2)
        Y(y(1,i),i)=1;
    end
end
%% 
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

