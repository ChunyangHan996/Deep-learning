function accuracy = ComputeAccuracySVM(X, y, W, b)
N = size(X,2);
accuracy = EvaluateClassifierSVM(X,W,b);
[C,I] = max(accuracy);
sum = 0;
for i = 1:N
    if(I(i)==y(i))
        sum = sum+1;
    end
end
accuracy = sum/N;
disp('accuracySVM=');
disp(accuracy);
end