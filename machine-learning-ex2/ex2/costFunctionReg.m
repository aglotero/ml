function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
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


for i =1:m
        x = sigmoid(sum(theta' .* X(i,:)));
        J = J + ((1 / m) * ( -y(i) * log(x)  - (1 - y(i))*log(1-x)));
end

regularization = 0;

for j = 2:n
	regularization = regularization + theta(j) ^ 2;	
end

regularization = (lambda / (2 * m))*regularization;

J = J + regularization;


%code for theta_0
for j = 1:n
        total = 0;
        for i = 1:m
                x = sigmoid(sum(theta' .* X(i,:)));
                total = total + ((x - y(i))*X(i,j));
        end
        grad(j) = (1/m) * total;
	if (j > 1)
		grad(j) = grad(j) +  (lambda / m) * theta(j) ;
	endif
end




% =============================================================

end
