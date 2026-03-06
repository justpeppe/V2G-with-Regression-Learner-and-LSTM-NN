function figureRegressionLearner = plotRegressionResults(yPredNorm, yNorm, params, target, timeVector)
% plotRegressionResults Plots the original vs predicted time series for Regression Learner models.
arguments
    yPredNorm double
    yNorm double
    params struct
    target
    timeVector datetime
end

if isstring(target)
    target = char(target);
end

yPredDenorm = (yPredNorm * params.(target).std) + params.(target).mean;
yRealDenorm = (yNorm * params.(target).std) + params.(target).mean;

yRealAligned = yRealDenorm(1:end-1);
yPredAligned = yPredDenorm(2:end);
timeVectorInput = timeVector(1:end-1);

figureRegressionLearner = figure;
plot(timeVectorInput, yRealAligned, "k-", "LineWidth", 1);
hold on;
plot(timeVectorInput, yPredAligned, "r--", "LineWidth", 1);
legend("Real Data", "Predicted Data");
title("Comparison Prediction vs Real De-normalized (Aligned)");
xlabel("Time");
ylabel("Energy (kWh)");
grid on;

end
