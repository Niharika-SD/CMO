# CMO

Code for CMO model introduced in https://link.springer.com/chapter/10.1007/978-3-030-20351-1_47

![CMO](https://github.com/Niharika-SD/CMO/blob/master/Images/Connectomics_and_Clinical_Severity_NL.PNG)
##Instructions

1. Open MATLAB (needs parallel computing toolbox)
2. Run main.m


#DATA ORGANIZATION

Main directory contents:

1. main.m #main script
2. altmin_run.m #alternating minimization loop
3. altmin_step.m #runs a single step of alternating minimization
4. update_basis.m #proximal gradient descent update
5. coefficient_update.m #trust region update for coeffs
6. constraint_updates.m #augmented lagrangian updates
7. error_compute.m #computes JNO error for a single iterate
8. quadestimate_Ctest.m #quadratic programming at test
9. compute_testscores.m #compute scores at test time
10. init_constraints.m #initialize constraint variables D and lamb
11. function_compute.m #compute the function in terms of coeffs and kernel
12. update_kernel.m #update for kernel matrix - implicitly for nl ridge regression weights

Folders:

~\Data\ADOS\data.mat
~\Data\ADOS\Outputs\Performance.mat
~\Data\ADOS\Outputs\models.mat

~\Data\SRS\data.mat
~\Data\SRS\Outputs\Performance.mat
~\Data\SRS\Outputs\models.mat

~\Data\Praxis\data.mat
~\Data\Praxis\Outputs\Performance.mat
~\Data\Praxis\Outputs\models.mat
   
