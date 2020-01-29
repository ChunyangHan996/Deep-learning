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