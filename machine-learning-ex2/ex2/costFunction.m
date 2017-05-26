function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples
total_theta = length(theta);

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%

for i =1:m
	x = sigmoid(sum(theta' .* X(i,:)));
	J = J + ((1 / m) * ( -y(i) * log(x)  - (1 - y(i))*log(1-x)));
end

for j = 1:total_theta
	total = 0;
	for i = 1:m
		x = sigmoid(sum(theta' .* X(i,:)));
		total = total + ((x - y(i))*X(i,j));
	end
	grad(j) = (1/m) * total;
end




% =============================================================

end
