function C_upd = coefficient_update(corr,B_upd,C,Y,D,lamb,alpha,lambda,lambda_2,sigma,w_p,p)
%%updates for coefficient- inputs to the kernel ridge regression

% pre-allocate
C_upd = zeros(size(C));

%decouple across patients
parfor m = 1:size(corr,1)
    
    %inout variables
    D_m = reshape(D(m,:,:),[size(D,2),size(D,3)]);
    lamb_m = reshape(lamb(m,:,:),[size(lamb,2),size(lamb,3)]);
    y = Y(m);
    
    % initial solution
    C_0 = C(:,m);
    
    fprintf('\n Patient %d',m)
    
    %bounds
    lb= zeros(size(C,1),1);
    ub= Inf*ones(size(C,1),1);
    
    %Calls to fmincon -trust region method
    options = optimoptions('fmincon','Algorithm','trust-region-reflective','SpecifyObjectiveGradient',true);
    C_m_old = fmincon([@(C_pat)obj_func_c(C_pat,C,B_upd,y,D_m,lamb_m,alpha,sigma,w_p,p,lambda,lambda_2)],C_0,[],[],[],[],lb,ub,[],options);
   
    
    C_upd(:,m) = C_m_old;
    
end

end     

function [F,G] = obj_func_c(C_pat,C,B_upd,y,D_m,lamb_m,alpha,sigma,w_p,p,lambda,lambda_2)

    y_val =0;

    grad_C_m= 2*lambda_2*C_pat;
      
    % const. term contribution
    grad_C_m = grad_C_m -diag((lamb_m'+D_m')*B_upd)+ C_pat.* diag(B_upd'*B_upd);
      
    % kernel function expansion contributions
    for i= 1:size(C,2)
           
        % function/gradient
        [F_i,J_i] = function_compute(C_pat,C(:,i),sigma,w_p,p);
    
        %output
        y_val= y_val + alpha(i)*F_i;
        
        % self terms
        grad_C_m = grad_C_m - 2*lambda*(alpha(i)*J_i).*y + 2*lambda*alpha(i)^2*J_i*F_i ;
       
        for k= 1:size(C,2)                                     
        % cross terms
            if (i~=k)
          
                [F_k,J_k] = function_compute(C_pat,C(:,k),sigma,w_p,p);
                grad_C_m = grad_C_m + lambda*alpha(i)*alpha(k)*(F_k*J_i+F_i*J_k);
            
            end
        
        end
    
    end
    
    %evaluate function and gradient
    F = lambda*(y-y_val).^2+ lambda_2*norm(C_pat,2)^2 + trace(lamb_m'*(D_m-B_upd*diag(C_pat)))+ 0.5* norm((D_m-B_upd*diag(C_pat)),'fro')^2;
    G = grad_C_m(:);

end