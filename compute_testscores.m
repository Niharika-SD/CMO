function est_Y = compute_test_scores(C_pat,C,K,Y,lambda_3,lambda,corr,sigma,w_p,p)
%%estimate Y based on the kernel and coeffcients

%pre-allocate
est_Y = zeros(size(corr,1),1);

%pre-compute alpha
alpha = pinv(K+(lambda_3/lambda)*eye(size(C,2)))*Y;        

%loop over patients during test
parfor n = 1: size(corr,1)
    %kernel definition
    for j = 1: size(C,2)
        

                [F_j,~] = function_compute(C_pat(:,n),C(:,j),sigma,w_p,p);
                est_Y(n) = est_Y(n) + alpha(j)*F_j;

    end
end

end