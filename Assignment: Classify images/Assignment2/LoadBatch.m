function [X, Y, y] = LoadBatch(filen)
    addpath ./Datasets/cifar-10-batches-mat/;
    A = load(filen);
    I = reshape(A.data', 32, 32, 3, 10000);
  
    X = double(reshape(I,3072, 10000))/255;% now data are between 0~1
    y = A.labels'+ones(1,size(A.labels',1));%1~10 instead of  0~9
    Y = zeros(10,size(y,2));

    for i = 1:size(X,2)
        Y(y(1,i),i)=1;
    end
end
