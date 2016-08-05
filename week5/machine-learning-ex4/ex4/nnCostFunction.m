function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

%transform y vector to binary positional vectors
eye_matrix = eye(num_labels);
y_positional = eye_matrix(y,:);


% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m

%add bias term to X
a1 = [ones(m, 1) X];

%feed forward first layer
z2 = a1 * Theta1';
a2 = sigmoid(z2); 

%feed forward through second layer
a2 = [ones(size(a2,1), 1) a2];
z3 = a2 * Theta2';
hypothesis = sigmoid(z3);


%skipping 1st column from regularization
Theta1_sq = Theta1(:,2:end).^2;
Theta2_sq = Theta2(:,2:end).^2;


cost = zeros(m, 1);
for i = 1 : m
	cost(i) = (y_positional(i,:) * log(hypothesis(i,:))') ...
				+ ((1 - y_positional(i,:)) * log(1 - hypothesis(i,:))');
end;


J = -(1/m) * sum(cost) ...
		+ (lambda/(2*m)) * ( sum(Theta1_sq(:)) + sum(Theta2_sq(:)) );


% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients

%create delta matrix...this will be of same size as of Theta (excluding bias term)
delta1 = zeros(size(Theta1));
delta2 = zeros(size(Theta2));
for i = 1 : m
	d3 = hypothesis(i,:) - y_positional(i,:);
	d2 = (d3 * Theta2(:,2:end)) .* sigmoidGradient(z2(i,:));
	delta1 = delta1 + d2' * a1(i,:);
	delta2 = delta2 + d3' * a2(i,:);
end

Theta1_grad = delta1./m;
Theta2_grad = delta2./m;

%
% Part 3: Implement regularization with the cost function and gradients.
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.

Theta1(:,1) = 0;
Theta2(:,1) = 0;

Theta1 = Theta1.*(lambda/m);
Theta2 = Theta2.*(lambda/m);

Theta1_grad = Theta1_grad + Theta1;
Theta2_grad = Theta2_grad + Theta2;

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];
end