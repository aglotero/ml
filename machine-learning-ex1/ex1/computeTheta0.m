function J = computeTheta0(X, y, theta)
m = length(y); % number of training examples

J = 0;

total = 0;

for i = 1:m
	total = total + ((theta(1) + theta(2) * X(i,2)) - y(i));
end

J = (1 /  m ) * total;

% =========================================================================

end
