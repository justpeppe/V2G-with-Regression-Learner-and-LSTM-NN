function indicators = get_indicators(YPredNorm, TData)

% De-normalizza
%YPred = YPredNorm * params.(target).dev_std + params.(target).media;
%TDataDenorm = TData * params.(target).dev_std + params.(target).media;

errors = TData - YPredNorm;
SSres = sum(errors.^2);
SStot = sum((TData - mean(TData)).^2);
indicators.RSquared = 1 - (SSres / SStot);

indicators.RMSE = sqrt(mean(errors.^2));
indicators.MSE = mean(errors.^2);

indicators.MAPE = mean(abs(errors ./ TData)) * 100;
indicators.MAE = mean(abs(errors));


end
