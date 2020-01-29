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