function indicators = getBestIndicators(infoNet)
% getBestIndicators Extracts the best training and validation metrics from network training info.
arguments
    infoNet
end

% Extract the validation loss history to find the best performing epoch
valLoss = infoNet.ValidationHistory.Loss;
validMask = ~isnan(valLoss); % Ignore epochs where validation wasn't calculated
validIndices = find(validMask);

if isempty(validIndices)
    warning("getBestIndicators: No valid validation loss found.");
    indicators = struct();
    return;
end

% Find the epoch with the lowest validation loss
[~, bestValIdxTemp] = min(valLoss(validIndices));
bestValIdx = validIndices(bestValIdxTemp);

% Extract validation metrics at the best epoch
indicators.Validation.Rsquared = infoNet.ValidationHistory.RSquared(bestValIdx);
indicators.Validation.RMSE = infoNet.ValidationHistory.RMSE(bestValIdx);
indicators.Validation.MSE = infoNet.ValidationHistory.MSE(bestValIdx);
indicators.Validation.MAPE = infoNet.ValidationHistory.MAPE(bestValIdx); % Already in percentage
indicators.Validation.MAE = infoNet.ValidationHistory.MAE(bestValIdx);
indicators.Validation.ValLoss = infoNet.ValidationHistory.Loss(bestValIdx);

% Find the epoch with the lowest training loss
bestTrainIdxTemp = find(infoNet.TrainingHistory.Loss == min(infoNet.TrainingHistory.Loss), 1, "first");
bestTrainIdx = bestTrainIdxTemp;

% Extract training metrics at the best epoch
indicators.Training.Rsquared = infoNet.TrainingHistory.RSquared(bestTrainIdx);
indicators.Training.RMSE = infoNet.TrainingHistory.RMSE(bestTrainIdx);
indicators.Training.MSE = infoNet.TrainingHistory.MSE(bestTrainIdx);
indicators.Training.MAPE = infoNet.TrainingHistory.MAPE(bestTrainIdx) * 100; % Convert to percentage
indicators.Training.MAE = infoNet.TrainingHistory.MAE(bestTrainIdx);
indicators.Training.TrainLoss = infoNet.TrainingHistory.Loss(bestTrainIdx);
end