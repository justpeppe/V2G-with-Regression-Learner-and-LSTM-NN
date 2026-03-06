function indicators = get_best_indicators(info)

    info_net = info;
    bestValIdx = find(info_net.ValidationHistory.Loss == min(info_net.ValidationHistory.Loss), 1, 'first');
    
    indicators.Validation.Rsquared = info_net.ValidationHistory.RSquared(bestValIdx);
    indicators.Validation.RMSE = info_net.ValidationHistory.RMSE(bestValIdx);
    indicators.Validation.MSE = info_net.ValidationHistory.MSE(bestValIdx);
    indicators.Validation.MAPE = info_net.ValidationHistory.MAPE(bestValIdx)*100;
    indicators.Validation.MAE = info_net.ValidationHistory.MAE(bestValIdx);
    indicators.Validation.ValLoss = info_net.ValidationHistory.Loss(bestValIdx);

    bestValIdx = find(info_net.TrainingHistory.Loss == min(info_net.TrainingHistory.Loss), 1, 'first');
    indicators.Training.Rsquared = info_net.TrainingHistory.RSquared(bestValIdx);
    indicators.Training.RMSE = info_net.TrainingHistory.RMSE(bestValIdx);
    indicators.Training.MSE = info_net.TrainingHistory.MSE(bestValIdx);
    indicators.Training.MAPE = info_net.TrainingHistory.MAPE(bestValIdx)*100;
    indicators.Training.MAE = info_net.TrainingHistory.MAE(bestValIdx);
    indicators.Training.ValLoss = info_net.TrainingHistory.Loss(bestValIdx);
end