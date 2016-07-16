function [J, grad] = costFunction(theta, X, y)

%initialize
J = 0;
grad = zeros(size(theta));
m = size(X,1);


%compute
logistic_prediction = sigmoid(X * theta);
J = -(1/m) * (log(logistic_prediction') * y + log(1-logistic_prediction')*(1-y));

grad = (1/m) * ((logistic_prediction - y)'*X)';

end
