[trainX, trainY, trainy] = LoadBatch('data_batch_1.mat');  %training
[validX, validY, validy] = LoadBatch('data_batch_2.mat');  %validation 
[testX, testY, testy] = LoadBatch('test_batch.mat');  %testing

d = size(trainX,1);
K = size(trainY,1);

rng(400);
W = 0 + 0.01 * randn(K,d); % Generate random number with expectation of 0, standard deviation of 0.01
b = 0 + 0.01 * randn(K,1);

n_epochs = 40; %the number of runs through the whole training set.
n_batch = 100; %the size of the mini-batches
eta = 0.01; %the learning rate
lambda = 1 ; % lambda=1;
GDparams = [n_batch, eta, n_epochs]; 

[Wstar, bstar] = MiniBatchGD(trainX, trainY, validX,validY,GDparams, W, b, lambda);
%[Wstar, bstar] = MiniBatchGD1(testX, testY,GDparams, W, b, lambda);
%ComputeAccuracy(trainX,trainy, Wstar, bstar);
%ComputeAccuracy(testX, testy,  Wstar, bstar);

for i=1:10
    im = reshape(Wstar(i, :), 32, 32, 3);
    s_im{i} = (im - min(im(:))) / (max(im(:)) - min(im(:))); 
    s_im{i} = permute(s_im{i}, [2, 1, 3]);
    subplot(1,10,i);
    title('n\_epoch=40,n\_batch=100,eta=.01,and lambda=1')
    imshow(s_im{i});
end

%%
function accuracy = ComputeAccuracy(X, y, W, b)
N = size(X,2);
accuracy = EvaluateClassifier(X,W,b);
[C,I] = max(accuracy);
sum = 0;
for i = 1:N
    if(I(i)==y(i))
        sum = sum+1;
    end
end
accuracy = sum/N;
disp('accuracy=');
disp(accuracy);
end

function J = ComputeCost(X, Y, W, b, lambda)
   second =( norm(W,2))^2 *lambda;
   sum = 0;
   N = size(X,2);
   for i = 1:N
       sum = sum + Crossentropy(X(:,i),Y(:,i),W,b);
   end
   sum = sum/N;
   J = sum + second;
end

function L = Crossentropy(X,Y,W,b)
p = EvaluateClassifier(X,W,b);
L = -log(Y'*p);
end

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

function [grad_b, grad_W] = ComputeGradsNum(X, Y, W, b, lambda, h)

no = size(W, 1);
d = size(X, 1);

grad_W = zeros(size(W));
grad_b = zeros(no, 1);

c = ComputeCost(X, Y, W, b, lambda);

for i=1:length(b)
    b_try = b;
    b_try(i) = b_try(i) + h;
    c2 = ComputeCost(X, Y, W, b_try, lambda);
    grad_b(i) = (c2-c) / h;
end

for i=1:numel(W)   
    
    W_try = W;
    W_try(i) = W_try(i) + h;
    c2 = ComputeCost(X, Y, W_try, b, lambda);
    
    grad_W(i) = (c2-c) / h;
end

function [grad_b, grad_W] = ComputeGradsNumSlow(X, Y, W, b, lambda, h)

no = size(W, 1);
d = size(X, 1);

grad_W = zeros(size(W));
grad_b = zeros(no, 1);

for i=1:length(b)
    b_try = b;
    b_try(i) = b_try(i) - h;
    c1 = ComputeCost(X, Y, W, b_try, lambda);
    b_try = b;
    b_try(i) = b_try(i) + h;
    c2 = ComputeCost(X, Y, W, b_try, lambda);
    grad_b(i) = (c2-c1) / (2*h);
end

for i=1:numel(W)
    
    W_try = W;
    W_try(i) = W_try(i) - h;
    c1 = ComputeCost(X, Y, W_try, b, lambda);
    
    W_try = W;
    W_try(i) = W_try(i) + h;
    c2 = ComputeCost(X, Y, W_try, b, lambda);
    
    grad_W(i) = (c2-c1) / (2*h);
end

function P=EvaluateClassifier(X,W,b)
    N = size(X,2);
    P=[];
    for i = 1:N
        s = W*X(:,i)+b;
        P = [P,softmax(s)];
    end
end

function P = softmax(s)
	P = exp(s);
	P = (1/sum(P))*P;
end

function [X, Y, y] = LoadBatch(filename)
addpath Datasets/cifar-10-batches-mat/;

addpath ./Datasets/cifar-10-batches-mat/;
    A = load(filename);
    I = reshape(A.data', 32, 32, 3, 10000);
    I = permute(I, [2, 1, 3, 4]);
    %montage (I(:,:,:,1:500),'Size',[25,20]);
    X = double(reshape(I,3072, 10000))/255;% now data are between 0~1
    y = A.labels'+ones(1,size(A.labels',1));%1~10 instead of  0~9
    Y = zeros(10,size(y,2));

    for i = 1:size(X,2)
        Y(y(1,i),i)=1;
    end
end

function [Wstar, bstar] = MiniBatchGD(X, Y, X1,Y1,GDparams, W, b, lambda)
    N = size(X,2);% return # of A's column
    n_batch = GDparams(1,1);
    eta = GDparams(1,2);
    n_epochs = GDparams(1,3);
    cost = ComputeCost(X, Y, W, b, lambda);
    trainingloss=[cost];
    validationloss=[cost];
    for i = 1:n_epoch
        for j=1:N/n_batch
            j_start = (j-1)*n_batch + 1;
            j_end = j*n_batch;
            inds = j_start:j_end;
            Xbatch = X(:, j_start:j_end);
            Ybatch = Y(:, j_start:j_end);
            P = EvaluateClassifier(Xbatch,W,b);
            [grad_W, grad_b] = ComputeGradients(Xbatch, Ybatch, P, W, lambda);
            W = W - eta * grad_W;
            b = b - eta * grad_b;
        end
        temp = ComputeCost(X, Y, W, b, lambda);
        temp1=ComputeCost(X1,Y1,W,b,lambda);
        trainingloss = [trainingloss;temp];% cost
        validationloss=[validationloss;temp1];
    end
      %for plot  
    plot(1:n_epochs+1, trainingloss,'g-',1:n_epochs+1,validationloss,'r-');
    title('n\_epoch=40,n\_batch=100,eta=.1,and lambda=0')
    xlabel('epoch');
    ylabel('loss');
    legend('Trainingloss','Validationloss')
    hold on;
    axis auto;
    
    Wstar = W;
    bstar = b;
end

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


%% bonus
[trainX, trainY, trainy] = LoadBatch('data_batch_1.mat');  %training
[validX, validY, validy] = LoadBatch('data_batch_2.mat');  %validation 
[testX, testY, testy] = LoadBatch('test_batch.mat');  %testing

d = size(trainX,1);
K = size(trainY,1);

rng(400);
W = 0 + 0.01 * randn(K,d); % Generate random number with expectation of 0, standard deviation of 0.01
b = 0 + 0.01 * randn(K,1);

n_epochs = 40; %the number of runs through the whole training set.
n_batch = 100; %the size of the mini-batches
eta = 0.01; %the learning rate
lambda = 0.1 ; % lambda=1;
GDparams = [n_batch, eta, n_epochs]; 

P=EvaluateClassifier(trainX,W,b);
[grad_W, grad_b] = ComputeGradients(trainX(1:100, 5:7), trainY(:,5:7), P(:,5:7), W(:,1:100), lambda);
%acc = ComputeAccuracy(trainX, trainy, W, b);

[Wstar, bstar] = MiniBatchGD1(trainX, trainY,GDparams, W, b, lambda);
ComputeAccuracy(testX,testy, Wstar, bstar);

Psvm = EvaluateClassifierSVM(trainX,W,b);
[grad_Wsvm, grad_bsvm] = ComputeGradientsSVM(trainX(1:100, 5:7), trainY(:,5:7), Psvm(:,5:7), W(:,1:100), lambda);
%acc_svm = ComputeAccuracySVM(trainX, trainy, W, b);
[Wstar, bstar] = MiniBatchGDSVM(trainX, trainY,GDparams, W, b, lambda);
ComputeAccuracySVM(testX,testy, Wstar, bstar);

