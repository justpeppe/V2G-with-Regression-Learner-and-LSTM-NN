function indicators = computeMetrics(yPredNorm, tDataNorm, params, target)
% computeMetrics Computes various regression metrics from normalized predictions and target data.
arguments
    yPredNorm double
    tDataNorm double
    params struct
    target (1,:) char
end

% Extract denormalization parameters for the target variable
stdVal  = params.(target).std;
meanVal = params.(target).mean;

% Denormalize predictions and actual targets back to their original scale (e.g., kWh)
yPred = yPredNorm .* stdVal + meanVal;
tData = tDataNorm .* stdVal + meanVal;

% Calculate raw errors (residuals)
errors = tData - yPred;

% Calculate R-squared (Coefficient of Determination)
ssRes  = sum(errors.^2); % Sum of squared residuals
ssTot  = sum((tData - mean(tData)).^2); % Total sum of squares
indicators.RSquared = 1 - (ssRes / ssTot);

% Calculate Root Mean Square Error (RMSE) and Mean Square Error (MSE)
indicators.RMSE = sqrt(mean(errors.^2));
indicators.MSE = mean(errors.^2);

% Calculate Mean Absolute Percentage Error (MAPE) and Mean Absolute Error (MAE)
indicators.MAPE = mean(abs(errors ./ tData)) * 100;
indicators.MAE = mean(abs(errors));

end
