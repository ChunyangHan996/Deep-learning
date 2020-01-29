function [dw1,dw2,db1,db2]= CompareGradient( X, Y, W, b, lambda, h,batch_size)
     T=3;
     P=EvaluateClassifier(X(:,1:batch_size),W,b);
     w={W{1}(:,1:batch_size);W{2}};
     [ga_W, ga_b] = ComputeGradients(X(1:batch_size, 1:3), Y(:,1:3), P(:,1:3), w, b, lambda);
     [gn_b, gn_W] = ComputeGradsNumSlow(X(1:batch_size, 1:3), Y(:,1:3), w, b, lambda,h);
     
    
    dw1=norm(ga_W{1}-gn_W{1})/max(eps,norm(ga_W{1}) + norm(gn_W{1}));
    dw2=norm(ga_W{2}-gn_W{2})/max(eps,norm(ga_W{2}) + norm(gn_W{2}));
    db1=norm(ga_b{1}-gn_b{1})/max(eps,norm(ga_b{1}) + norm(gn_b{1}));
    db2=norm(ga_b{2}-gn_b{2})/max(eps,norm(ga_b{2}) + norm(gn_b{2}));
    

    disp(dw1);
    disp(dw2);
    disp(db1);
    disp(db2);
end