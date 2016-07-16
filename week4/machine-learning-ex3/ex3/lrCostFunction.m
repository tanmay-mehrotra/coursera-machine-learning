function [J, grad] = lrCostFunction(theta, X, y, lambda)
%initialize
J = 0;
grad = zeros(size(theta));
m = size(X,1);


%compute
logistic_prediction = sigmoid(X * theta);
J = -(1/m)*(log(logistic_prediction') * y + log(1-logistic_prediction') * (1-y)) + (lambda/(2*m))*(theta(2:end)'* theta(2:end));

temp = (1/m) * ((logistic_prediction - y)' * X)';

grad = temp + ((lambda/m) * theta);

%skip regularized component from grad(0)
grad(1) = temp(1);

end
