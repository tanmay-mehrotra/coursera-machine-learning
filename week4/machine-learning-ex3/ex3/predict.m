function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);
X = [ones(m, 1) X];


%calculating activation points for hidden layer aka 2nd layer
%z2 denotes z supercript 2
z2 = Theta1 * X';
a2 = sigmoid(z2);

%calculating activation points for output layer 
%add bias to a2
a2 = [ones(1, size(a2,2)); a2];
z3 = Theta2 * a2;
a3 = sigmoid(z3);

[value,index] = max(a3,[],1);

p = index';

end
