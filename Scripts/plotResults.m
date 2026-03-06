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

% De-normalize the predictions and the actual target data back to the original scale
yReal = yNorm .* params.(target).std + params.(target).mean;
yPred = yPredNorm .* params.(target).std + params.(target).mean;

% LSTM predictions map exactly 1-to-1 with the test sequences, so no shifting is needed
timeVectorInput = timeVector;
yRealAligned = yReal;
yPredAligned = yPred;

% ── 1. Create a logical index array instead of time array ───────────────
numPoints = length(yRealAligned);
xIdx = 1:numPoints;

% ── 2. Find Boundaries Between Days ──────────────────────────────────────
% We want to draw a vertical line when the date changes
dayStamps = dateshift(timeVectorInput, "start", "day");
uniqueDays = unique(dayStamps);
numDays = numel(uniqueDays);

% Find the indices where each day starts and ends
dayStartIdx = zeros(numDays, 1);
dayEndIdx   = zeros(numDays, 1);
dayMidIdx   = zeros(numDays, 1);
dayLabels   = strings(numDays, 1);

for k = 1:numDays
    mask = (dayStamps == uniqueDays(k));
    idxs = find(mask);
    dayStartIdx(k) = idxs(1);
    dayEndIdx(k)   = idxs(end);
    dayMidIdx(k)   = round(mean(idxs)); % Center point for X-axis label
    dayLabels(k)   = string(uniqueDays(k), "yyyy-MM-dd");
end

% ── 3. Plotting ──────────────────────────────────────────────────────────
figureOut = figure;
plot(xIdx, yRealAligned, "k.-", MarkerSize=10, LineWidth=1.2, DisplayName="Real Data");
hold on;
plot(xIdx, yPredAligned, "r.-", MarkerSize=10, LineWidth=1.2, DisplayName="Predicted Data");

% Draw vertical separators between days
yLimits = ylim;
for k = 2:numDays
    line([dayStartIdx(k)-0.5, dayStartIdx(k)-0.5], yLimits, "Color", [0.5 0.5 0.5], "LineStyle", "--", "HandleVisibility", "off");
end

% Format X-Axis to show exactly the dates instead of arbitrary indices
xticks(dayMidIdx);
xticklabels(dayLabels);
xtickangle(45); % Tilt for readability

legend("Real Data", "Predicted Data");
title("Comparison Prediction vs Real De-normalized (Side-by-Side)");
xlabel("Selected Days");
ylabel("Energy (kWh)");
grid on;
hold off;
end
