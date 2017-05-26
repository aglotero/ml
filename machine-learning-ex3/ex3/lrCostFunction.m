function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples
n = length(theta);

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Hint: The computation of the cost function and gradients can be
%       efficiently vectorized. For example, consider the computation
%
%           sigmoid(X * theta)
%
%       Each row of the resulting matrix will contain the value of the
%       prediction for that example. You can make use of this to vectorize
%       the cost function and gradient computations. 
%
% Hint: When computing the gradient of the regularized cost function, 
%       there're many possible vectorized solutions, but one solution
%       looks like:
%           grad = (unregularized gradient for logistic regression)
%           temp = theta; 
%           temp(1) = 0;   % because we don't add anything for j = 0  
%           grad = grad + YOUR_CODE_HERE (using the temp variable)
%


%for i =1:m
%        x = sigmoid(sum(theta' .* X(i,:)));
%        J = J + ((1 / m) * ( -y(i) * log(x)  - (1 - y(i))*log(1-x)));
%end

%regularization = 0;

%for j = 2:n
%	regularization = regularization + theta(j) ^ 2;	
%end

%regularization = (lambda / (2 * m))*regularization;

%J = J + regularization;


%code for theta_0
%for j = 1:n
%        total = 0;
%        for i = 1:m
%                x = sigmoid(sum(theta' .* X(i,:)));
%                total = total + ((x - y(i))*X(i,j));
%        end
%        grad(j) = (1/m) * total;
%	if (j > 1)
%		grad(j) = grad(j) +  (lambda / m) * theta(j) ;
%	endif
%end

% calculate hypothesis
h = sigmoid(X*theta);

% regularize theta by removing first value
theta_reg = [0;theta(2:end, :);];

J = (1/m)*(-y'* log(h) - (1 - y)'*log(1-h))+(lambda/(2*m))*theta_reg'*theta_reg;

grad = (1/m)*(X'*(h-y)+lambda*theta_reg);





% =============================================================

grad = grad(:);

end
