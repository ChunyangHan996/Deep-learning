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
