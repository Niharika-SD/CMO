function K = update_kernel(C,sigma,w_p,p)
%Update the kernel matrix K 

N = size(C,2);
K = zeros(N);

%pointwise update
for i= 1:N
    for j=i:N
      
        c_i = C(:,i);
        c_j = C(:,j);
        
        [K(i,j),~] = function_compute(c_j,c_i,sigma,w_p,p);
        K(j,i) = K(i,j);
        
    end
end


end