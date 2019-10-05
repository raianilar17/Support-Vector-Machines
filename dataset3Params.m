function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== IMPORTANT CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

I_C = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
I_sigma = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];

error = zeros(size(I_C), size(I_sigma));

for i = 1:length(I_C),
  for j = 1: length(I_sigma),
    model= svmTrain(X, y, I_C(i), @(x1, x2) gaussianKernel(x1, x2, I_sigma(j)));
    predictions = svmPredict(model, Xval);
    error(i,j) = mean(double(predictions ~= yval));
  end;
end;
    
[c_v c_idx] = min(min(error, [], 2)); %% row
[sig_v sig_idx] = min(min(error));   %% column

if c_v == sig_v,
  C = I_C(c_idx);
  sigma = I_sigma(sig_idx);
end;


% =========================================================================

end
