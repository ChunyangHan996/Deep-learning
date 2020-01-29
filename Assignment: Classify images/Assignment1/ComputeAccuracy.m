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