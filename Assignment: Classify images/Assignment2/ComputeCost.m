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