function indicators = computeMetrics(yPredNorm, tDataNorm, params, target)
% computeMetrics Computes various regression metrics from normalized predictions and target data.
arguments
    yPredNorm double
    tDataNorm double
    params struct
    target (1,:) char
end

stdVal  = params.(target).std;
meanVal = params.(target).mean;

yPred = yPredNorm .* stdVal + meanVal;
tData = tDataNorm .* stdVal + meanVal;

errors = tData - yPred;
ssRes  = sum(errors.^2);
ssTot  = sum((tData - mean(tData)).^2);
indicators.RSquared = 1 - (ssRes / ssTot);

indicators.RMSE = sqrt(mean(errors.^2));
indicators.MSE = mean(errors.^2);

indicators.MAPE = mean(abs(errors ./ tData)) * 100;
indicators.MAE = mean(abs(errors));

end
