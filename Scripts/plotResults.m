function figureOut = plotResults(yPredNorm, yNorm, params, target, timeVector)
% plotResults Plots the original vs predicted time series for LSTM models.
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

yReal = yNorm .* params.(target).std + params.(target).mean;
yPred = yPredNorm .* params.(target).std + params.(target).mean;

timeVectorInput = timeVector;
yRealAligned = yReal;
yPredAligned = yPred;

figureOut = figure;
plot(timeVectorInput, yRealAligned, "k-", "LineWidth", 1);
hold on;
plot(timeVectorInput, yPredAligned, "r--", "LineWidth", 1);
legend("Real Data", "Predicted Data");
title("Comparison Prediction vs Real De-normalized (Aligned)");
xlabel("Time");
ylabel("Energy (kWh)");
grid on;
hold off;
end
