function Jsvm = ComputeCostSVM(X, Y, W, b, lambda)
   second = lambda * norm(W) * norm(W);
   sum = 0;
   N = size(X,2);
   for i = 1:N
       label = 0;
       for j = 1:10
           if Y(j,i)==1
               label = j;
               break
           end
       end
       sum = sum + l_cross(X(:,i),Y(:,i),W,b,label);
   end
   sum = sum/N;
   Jsvm = sum + second;
end

function L = l_cross(x,y,W,b,truelabel)
p = EvaluateClassifierSVM(x,W,b);
M = size(p,2);
N = size(p,1); %10 property
L = 0;
count = 0;
for i = 1:N
    if i==truelabel
        continue
    end
    if (p(i,1)-p(truelabel,1)+1) > 0
        count = count + 1;
        L = L + p(i,1)-p(truelabel,1) + 1;
    end
end
end