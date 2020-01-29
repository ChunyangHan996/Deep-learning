function Psvm=EvaluateClassifierSVM(X,W,b)
% N = size(X,2);
% Psvm=[];
% for i = 1:N
%     s = W*X(:,i)+b;
%     Psvm = [Psvm,s];
% end
Psvm = W*X+b;
end