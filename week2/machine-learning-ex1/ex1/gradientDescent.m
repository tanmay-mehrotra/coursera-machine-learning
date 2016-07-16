function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)

    % Initialize some useful values
    m = length(y);
    J_history = zeros(num_iters, 1);
    no_of_features = size(X,2);

    for iter = 1:num_iters
        current_hypothesis  = (X*theta - y);
        for feature = 1:no_of_features
            current_feature = X(:,feature);
            current_theta = theta(feature);
            current_theta = current_theta - (alpha/m)*(current_hypothesis' * current_feature);
            theta(feature) = current_theta;
        end
        J_history(iter) = computeCost(X, y, theta)
    end
end
