function [bestEpsilon bestF1] = selectThreshold(yval, pval)
%SELECTTHRESHOLD Find the best threshold (epsilon) to use for selecting
%outliers
%   [bestEpsilon bestF1] = SELECTTHRESHOLD(yval, pval) finds the best
%   threshold to use for selecting outliers based on the results from a
%   validation set (pval) and the ground truth (yval).
%

bestEpsilon = 0;
bestF1 = 0;
F1 = 0;

m = size(yval,1);

stepsize = (max(pval) - min(pval)) / 1000;
for epsilon = min(pval):stepsize:max(pval)
    
    tp = 0; fp = 0; tn = 0; fn = 0;
    predictions = (pval < epsilon);

    for i = 1 : m
        if(predictions(i) == 1 & yval(i) == 1)
            tp = tp + 1;
        elseif(predictions(i) == 1 & yval(i) == 0)
            fp = fp + 1;
        elseif(predictions(i) == 0 & yval(i) == 1)
            fn = fn + 1;
        else
            tn = tn +1;
        end
    end

    prec = tp/(tp + fp);
    recall = tp/(tp + fn);

    F1 = (2 * prec * recall) / (prec + recall);

    % =============================================================

    if F1 > bestF1
       bestF1 = F1;
       bestEpsilon = epsilon;
    end
end

end
