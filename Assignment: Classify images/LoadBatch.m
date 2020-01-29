function [X, Y, y] = LoadBatch(filename)
addpath Datasets/cifar-10-batches-mat/;

addpath ./Datasets/cifar-10-batches-mat/;
    A = load(filename);
    I = reshape(A.data', 32, 32, 3, 10000);
    I = permute(I, [2, 1, 3, 4]);
    %montage (I(:,:,:,1:500),'Size',[25,20]);
    X = double(reshape(I,3072, 10000))/255;% now data are between 0~1
    y = A.labels'+ones(1,size(A.labels',1));%1~10 instead of  0~9
    Y = zeros(10,size(y,2));

    for i = 1:size(X,2)
        Y(y(1,i),i)=1;
    end
end
