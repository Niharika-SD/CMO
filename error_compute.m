function err = error_compute(corr,B,C,C_bar,K,Y,D,lamb,lambda,lambda_1,lambda_2,lambda_3,sigma,w_p,p,flag)
%%Computes the error at the current interation given the values of the iterates at the instant

%pre-allocate
fit_err = 0;
const_err =0 ;
aug_lag_err =0;
est_Y = zeros(size(Y));
err_W_reg = 0;

for n = 1:size(corr,1)
    
    %input variables  
    D_n = reshape(D(n,:,:),[size(D,2),size(D,3)]);
    lamb_n = reshape(lamb(n,:,:),[size(lamb,2),size(lamb,3)]);
     
    %compute correlation fit
    X = reshape((corr(n,:,:)),[size(corr,2),size(corr,3)]) - D_n*B'; 
    fit_err = fit_err + norm(X,'fro').^2;
    
    %compute lagrangian + reg. error
    const_err = const_err + trace(lamb_n'*(D_n-B*diag(C(:,n))));
    aug_lag_err = aug_lag_err +0.5*norm((D_n-B*diag(C(:,n))),'fro').^2;
    
    %Compute error for kernel ridge reg. flag      
    alpha = pinv(K +(lambda_3/lambda)*eye(size(C,2)))*Y;        
        
     for j = 1:size(corr,1)
       
        % saves computation
        if (flag)
            
            %re-compute only if flag is 1 
            [F_j,~] = function_compute(C(:,n),C_bar(:,j),sigma,w_p,p);
        
        else
            
            %else use kernel value
            F_j = K(n,j);
        
        end
        
        %update estimate
        est_Y(n) = est_Y(n) + alpha(j)*F_j;
                      
        % update regularization term
        err_W_reg = err_W_reg + alpha(n)*alpha(j)* K(n,j);
        
     end
        
end

%total error
err = fit_err + const_err + aug_lag_err + lambda_1* norm(B,1) + lambda* norm(est_Y-Y,'fro').^2 + lambda_2* norm(C,'fro').^2+ lambda_3*err_W_reg;

end