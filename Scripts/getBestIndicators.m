function indicators = getBestIndicators(infoNet)
% getBestIndicators Extracts the best training and validation metrics from network training info.
arguments
    infoNet
end

valLoss = infoNet.ValidationHistory.Loss;
validMask = ~isnan(valLoss);
validIndices = find(validMask);

if isempty(validIndices)
    warning("getBestIndicators: No valid validation loss found.");
    indicators = struct();
    return;
end

[~, bestValIdxTemp] = min(valLoss(validIndices));
bestValIdx = validIndices(bestValIdxTemp);

indicators.Validation.Rsquared = infoNet.ValidationHistory.RSquared(bestValIdx);
indicators.Validation.RMSE = infoNet.ValidationHistory.RMSE(bestValIdx);
indicators.Validation.MSE = infoNet.ValidationHistory.MSE(bestValIdx);
indicators.Validation.MAPE = infoNet.ValidationHistory.MAPE(bestValIdx) * 100;
indicators.Validation.MAE = infoNet.ValidationHistory.MAE(bestValIdx);
indicators.Validation.ValLoss = infoNet.ValidationHistory.Loss(bestValIdx);

bestTrainIdxTemp = find(infoNet.TrainingHistory.Loss == min(infoNet.TrainingHistory.Loss), 1, "first");
bestTrainIdx = bestTrainIdxTemp;

indicators.Training.Rsquared = infoNet.TrainingHistory.RSquared(bestTrainIdx);
indicators.Training.RMSE = infoNet.TrainingHistory.RMSE(bestTrainIdx);
indicators.Training.MSE = infoNet.TrainingHistory.MSE(bestTrainIdx);
indicators.Training.MAPE = infoNet.TrainingHistory.MAPE(bestTrainIdx) * 100;
indicators.Training.MAE = infoNet.TrainingHistory.MAE(bestTrainIdx);
indicators.Training.TrainLoss = infoNet.TrainingHistory.Loss(bestTrainIdx);
end