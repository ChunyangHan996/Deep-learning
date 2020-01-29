clear;
book_fname = 'goblet_book.txt';
fid = fopen(book_fname,'r'); 
book_data = fscanf(fid,'%c');
fclose(fid);

sig  = 0.01;
seq_length = 25;%read 25 chars each time
eta = 0.1;
m = 100;%hidden nodes

alph = unique(book_data);

char_to_ind = containers.Map('KeyType','char','ValueType','int32');
ind_to_char = containers.Map('KeyType','int32','ValueType','char');
K = size(alph,2);
for i=1:K
    char_to_ind(alph(1,i)) = i;
    ind_to_char(i) = alph(1,i);
end

% rng(400);
RNN.U = randn(m, K)*sig;
RNN.V = randn(K, m)*sig;
RNN.W = randn(m, m)*sig;
RNN.b = zeros(m,1);
RNN.c = zeros(K,1);
h_init = zeros(m, 1);

M.U = 0;
M.V = 0;
M.W = 0;
M.b = 0;
M.c = 0;

smooth_loss = 0;
count = floor(size(book_data,2)/seq_length);
p = 0;

%%%%%% 1. Compare the Analytic & Numerical Gradients
%     X_chars = book_data(1:seq_length);
%     Y_chars = book_data(2:seq_length+1);
%     X = convert(X_chars,K,char_to_ind);
%     Y = convert(Y_chars,K,char_to_ind);
%     [grad_a,finalh] = ComputeGradient(X,Y,RNN,h_init);
%     h = 1e-4;
%     grad_n = ComputeGradsNum(X, Y, RNN, h);
%     
%     dw=norm(grad_a.W-grad_n.W)/max(norm(grad_a.W),norm(grad_n.W));
%     du=norm(grad_a.U-grad_n.U)/max(norm(grad_a.U),norm(grad_n.U));
%     dv=norm(grad_a.V-grad_n.V)/max(norm(grad_a.V),norm(grad_n.V));
%     db=norm(grad_a.b-grad_n.b)/max(norm(grad_a.b),norm(grad_n.b));
%     dc=norm(grad_a.c-grad_n.c)/max(norm(grad_a.c),norm(grad_n.c));
%  
%     display(dw);
%     display(du);
%     display(dv);
%     display(db);
%     display(dc);

dispsm_loss = [];

for k =1:132905 %2. times to iterate %10000000
    e = mod(k,count)*seq_length + 1;
    if floor(k/count) ~= p
        p = floor(k/count);
        h_init = zeros(m, 1);
    end
    if mod(k,1000)==0
        disp(k);
        disp(smooth_loss);
        dispstr_length = 1000;
        x_1 = zeros(K,1)*sig;
        x_1(char_to_ind(book_data(1,e))) = 1;
        YY = synthesize(h_init,x_1,dispstr_length,RNN,K);
        dispstr = [];
        for i = 1:dispstr_length
            dispstr = [dispstr,ind_to_char(vec2ind(YY(:,i)))];
        end
        display(dispstr);
    end
    %%%%%%%%%%%%% 2. plot at every 5000 update time 
     if mod(k,5000)==0
         %disp(k);
           dispsm_loss = [dispsm_loss;smooth_loss];
     end
    X_chars = book_data(e:e+seq_length-1);
    Y_chars = book_data(e+1:e+seq_length);
    X = convert(X_chars,K,char_to_ind);
    Y = convert(Y_chars,K,char_to_ind);
    [grad,h] = ComputeGradient(X,Y,RNN,h_init);
    if k == 1
        smooth_loss = ComputeLoss(X, Y, RNN, h_init);
    else
        smooth_loss = 0.999 * smooth_loss + 0.001*ComputeLoss(X, Y, RNN, h_init);
    end
    
    h_init = h;
    for f = fieldnames(RNN)'
        grad.(f{1}) = max(min(grad.(f{1}), 5), -5);
        tmp = grad.(f{1}).^2;
        M.(f{1}) = M.(f{1}) + tmp;
        RNN.(f{1}) = RNN.(f{1}) - eta*grad.(f{1})./((M.(f{1})+1e-8).^(0.5));
    end
    
end
%%%%%%%%%%%%%%%% 2. plot about smooth loss 
% sizes = size(dispsm_loss,1);
%     plot(1:sizes, dispsm_loss,'b');
%     title('Smooth loss for 3 Epoches, ');
%     xlabel('iterate times');
%     ylabel('smooth loss');

%%%
dispstr_length = 1000;
h_init = rand(m, 1)*sig;
x_1 = zeros(K,1)*sig;
x_1(floor(rand()*83),1) = 1;
Y = synthesize(h_init,x_1,dispstr_length,RNN,K);
dispstr = [];

for i = 1:dispstr_length
    dispstr = [dispstr,ind_to_char(vec2ind(Y(:,i)))];
end
display(dispstr);

function [K] = convert(s,M,char_to_ind)
    K = zeros(M,size(s,2));
    for i=1:size(s,2)
        K(char_to_ind(s(:,i)),i)=1;
    end
end

function [grad,finalh] = ComputeGradient(X,Y,RNN,hpre)
    K = size(RNN.c,1);
    L = size(RNN.W,1);
    M = size(X,2);
   
    grad.U = zeros(L, K);
    grad.V = zeros(K, L);
    grad.W = zeros(L, L);
    grad.b = zeros(L,1);
    grad.c = zeros(K,1);
   
    grad_o = zeros(M,K);
    a = zeros(L,M);
    h = zeros(L , M+1);
    h(:,1) = hpre;
    
    for i=1:M
        a(:,i) = RNN.W*h(:,i)+RNN.U*X(:,i) + RNN.b;
        h(:,i+1) = tanh(a(:,i));
        o = RNN.V*h(:,i+1) + RNN.c;
        p = softmax(o);
        grad_o(i,:) = -(Y(:,i)-p)';
        grad.V = grad.V + grad_o(i,:)'*h(:,i+1)';
        grad.c = grad.c + grad_o(i,:)';
    end
    grad_h = zeros(M,L);
    grad_a = zeros(M,L);
    grad_h(M,:) = grad_o(M,:)*RNN.V;
    grad_a(M,:) = grad_h(M,:)*diag(1 - tanh(a(:,M)).^2);
    for i = M-1:-1:1
        grad_h(i,:) = grad_o(i,:)*RNN.V + grad_a(i+1,:)*RNN.W;
        grad_a(i,:) = grad_h(i,:)*diag(1 - tanh(a(:,i)).^2);
    end
    for i = 1:M
        grad.W = grad.W + grad_a(i,:)'*h(:,i)';
        grad.U = grad.U + grad_a(i,:)'*X(:,i)';
        grad.b = grad.b + grad_a(i,:)';
    end
    finalh = h(:,M+1);
end

function[Y] = synthesize(h_pre,x_pre,n,RNN,K)
    Y = zeros(K,n);
    for i=1:n
        a = RNN.W*h_pre+RNN.U*x_pre + RNN.b;
        h = tanh(a);
        o = RNN.V*h + RNN.c;
        p = softmax(o);
        cp = cumsum(p);
        a = rand;
        ixs = find(cp-a >0);
        ii = ixs(1);
        Y(ii,i)=1;
        h_pre = h;
        x_pre = zeros(K, 1);
        x_pre(ii,1) = 1;
    end
end

function num_grads = ComputeGradsNum(X, Y, RNN, h)

    for f = fieldnames(RNN)'
        disp('Computing numerical gradient for')
        disp(['Field name: ' f{1} ]);
        num_grads.(f{1}) = ComputeGradNumSlow(X, Y, f{1}, RNN, h);
    end
end

function grad = ComputeGradNumSlow(X, Y, f, RNN, h)
    n = numel(RNN.(f));
    grad = zeros(size(RNN.(f)));
    hprev = zeros(size(RNN.W, 1), 1);
    for i=1:n
    RNN_try = RNN;
    RNN_try.(f)(i) = RNN.(f)(i) - h;
    l1 = ComputeLoss(X, Y, RNN_try, hprev);
    RNN_try.(f)(i) = RNN.(f)(i) + h;
    l2 = ComputeLoss(X, Y, RNN_try, hprev);
    grad(i) = (l2-l1)/(2*h);
end

end
function [loss] = ComputeLoss(X, Y, RNN, hprev)
    loss = 0;
    for i = 1:size(X,2)
        a = RNN.W*hprev+RNN.U*X(:,i) + RNN.b;
        h = tanh(a);
        o = RNN.V*h + RNN.c;
        p = softmax(o);
        hprev = h;
        loss = loss - log(Y(:,i)'*p);
    end
end

