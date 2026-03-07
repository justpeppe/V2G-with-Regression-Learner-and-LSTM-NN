%% RegressionLearner_Predictions.m
% Cross-zone generalization test of the most recently trained Regression Learner model.
%
% For every available zone, this script:
%   1. Loads zone data and normalizes it using the zone's own z-score parameters.
%   2. Generates feature lags using createRegressionLags.
%   3. Evaluates the trained regression model using predict().q
%   4. Produces diagnostic figures (Scatter, Residuals, Hourly Error, Time Series) per zone.
%   5. Prints a comprehensive metric table and an automated performance analysis.

%% User Configuration Parameters
% 1. Target Zones: List of zone IDs to evaluate (e.g., [8, 9, 10, 11])
targetZones = [8, 9, 10, 11];

% 2. Test Day Mode:
%    - 'representative' : Selects the "best" days from each cluster (default)
%    - 'all'            : Evaluates on the entire available time series
%    - 'custom'         : Evaluates on specific dates provided in 'customDates'
testMode = 'all';

% 3. Custom Dates: Only used if testMode = 'custom'
%    Format: datetime array, e.g., [datetime(2023,02,15), datetime(2023,02,16)]
customDates = [];

%% Setup
clearvars -except targetZones testMode customDates;
root = fileparts(mfilename('fullpath'));
addpath(root + "\Scripts");

%% Load Most Recent Regression Model
sessioniDir = fullfile(root, "Sessioni");
dateFiles   = dir(fullfile(sessioniDir, "**", "Models_*.mat"));

if isempty(dateFiles)
    error("No saved models found in Sessioni/. Run RegressionLearner.m first.");
end
[~, idx] = sort([dateFiles.datenum], "descend");
latestFile = fullfile(dateFiles(idx(1)).folder, dateFiles(idx(1)).name);
fprintf("Model file: %s\n\n", latestFile);

loaded  = load(latestFile, "models");
mdls    = loaded.models;

allNames = fieldnames(mdls);
isRegModel = startsWith(allNames, "reg", "IgnoreCase", true);
regNames = allNames(isRegModel);

if isempty(regNames)
    error("No Regression Learner models found in the latest file: %s", latestFile);
end

netName = regNames{end};
m       = mdls.(netName);
cfg     = m.config;

fprintf("Loaded %s: %s  (trained on Zone %d)\n",  m.modelType, netName, m.zoneId);
fprintf("Features: %s\n", strjoin(cfg.predictors, ", "));
fprintf("NumLags:  %d\n\n", cfg.numLags);

%% Setup Report Directory & Diary
[netPath, ~] = fileparts(latestFile);
reportDir = fullfile(netPath, "Report_" + netName);
if ~exist(reportDir, "dir")
    mkdir(reportDir);
end
if ~exist(fullfile(reportDir, "fig"), "dir")
    mkdir(fullfile(reportDir, "fig"));
end
diary(fullfile(reportDir, "Predictions_Log.txt"));

%% Zone Loop
zoneIds = targetZones;

% Pre-allocate summary
summaryZone  = zeros(numel(zoneIds), 1);
summaryR2    = NaN(numel(zoneIds), 1);
summaryRMSE  = NaN(numel(zoneIds), 1);
summaryMAE   = NaN(numel(zoneIds), 1);
summaryR2Pers = NaN(numel(zoneIds), 1);
summaryOK    = false(numel(zoneIds), 1);

for zi = 1:numel(zoneIds)
    zId = zoneIds(zi);
    summaryZone(zi) = zId;

    fprintf("\n--- ZONE %d ---\n", zId);

    try
        %% 1. Load & enrich data
        datas = loadZoneData(root, zId);
        datas.hour_sin = sin(2 * pi * hour(datas.time_vector) / 24);
        datas.hour_cos = cos(2 * pi * hour(datas.time_vector) / 24);
        datas.day_sin  = sin(2 * pi * day(datas.time_vector, "dayofweek") / 7);
        datas.day_cos  = cos(2 * pi * day(datas.time_vector, "dayofweek") / 7);

        %% 2. Zone-specific normalization (calculates mu/sigma per zone)
        colsNeeded = unique([cfg.predictors, cfg.target, "time_vector"], "stable");
        colsNeeded = colsNeeded(ismember(colsNeeded, datas.Properties.VariableNames));
        datas      = datas(:, colsNeeded);

        colsToNorm = unique([cfg.predictors, cfg.target], "stable");
        colsToNorm = colsToNorm(ismember(colsToNorm, datas.Properties.VariableNames));

        normPs   = struct();
        datasNorm = datas;
        for c = 1:numel(colsToNorm)
            col = colsToNorm(c);
            v   = datas.(col);
            mu  = mean(v, "omitnan");
            sg  = std(v, 0, "omitnan");
            if sg == 0 || isnan(sg); sg = 1; end
            datasNorm.(col) = (v - mu) ./ sg;
            normPs.(matlab.lang.makeValidName(col)).mean = mu;
            normPs.(matlab.lang.makeValidName(col)).std  = sg;
        end
        datasNorm.time_vector = datas.time_vector;

        %% 3. Select test days
        if strcmpi(testMode, 'representative')
            [~, testDays] = selectRepresentativeDays(datas);
            testDayStamps = dateshift(datas.time_vector, "start", "day");
            testDaysToKeep = unique(dateshift(testDays, "start", "day"));
            testMask      = ismember(testDayStamps, testDaysToKeep);
            fprintf("Mode: REPRESENTATIVE (Selected days: %s)\n", strjoin(string(testDaysToKeep, "yyyy-MM-dd"), ", "));

        elseif strcmpi(testMode, 'all')
            testMask = true(size(datas.time_vector));
            testDaysToKeep = unique(dateshift(datas.time_vector, "start", "day"));
            fprintf("Mode: ALL DATA (Total days: %d)\n", numel(testDaysToKeep));

        elseif strcmpi(testMode, 'custom')
            if isempty(customDates)
                error("testMode is 'custom' but customDates is empty.");
            end
            testDayStamps = dateshift(datas.time_vector, "start", "day");
            testDaysToKeep = unique(dateshift(customDates, "start", "day"));
            testMask      = ismember(testDayStamps, testDaysToKeep);
            fprintf("Mode: CUSTOM (User-defined days: %s)\n", strjoin(string(testDaysToKeep, "yyyy-MM-dd"), ", "));

        else
            error("Invalid testMode: %s", testMode);
        end

        testData = datasNorm(testMask, :);

        %% 4. Create Lags and Predict
        testLag  = createRegressionLags(testData, cfg.numLags, cfg.predictors, char(cfg.target));

        featureCols = testLag.Properties.VariableNames;
        featureCols = featureCols(~strcmp(featureCols, "time_vector") & ~strcmp(featureCols, char(cfg.target)));

        XTest  = table2array(testLag(:, featureCols));
        yTestNorm  = testLag.(char(cfg.target));
        timeVectorTestMat = testLag.time_vector;

        if isempty(XTest)
            error("No test samples generated for zone %d after lagging.", zId);
        end

        yPredNorm = predict(m.mdl, XTest);

        % Denormalize
        targetField = matlab.lang.makeValidName(char(cfg.target));
        mu  = normPs.(targetField).mean;
        sg  = normPs.(targetField).std;

        yPredReal = double(yPredNorm) .* sg + mu;
        yTestReal = double(yTestNorm) .* sg + mu;

        %% 5. Metrics
        errors  = yTestReal - yPredReal;
        ssRes   = sum(errors.^2);
        ssTot   = sum((yTestReal - mean(yTestReal)).^2);
        r2      = 1 - ssRes / ssTot;
        rmse    = sqrt(mean(errors.^2));
        mae     = mean(abs(errors));
        nonZero = yTestReal ~= 0;
        mape    = mean(abs(errors(nonZero) ./ yTestReal(nonZero))) * 100;

        % Persistence baseline
        yPers        = yTestReal(1:end-1);
        yTestShifted = yTestReal(2:end);
        ssResPers    = sum((yTestShifted - yPers).^2);
        ssTotPers    = sum((yTestShifted - mean(yTestShifted)).^2);
        r2Pers       = 1 - ssResPers / ssTotPers;
        rmsePers     = sqrt(mean((yTestShifted - yPers).^2));

        summaryR2(zi)    = r2;
        summaryRMSE(zi)  = rmse;
        summaryMAE(zi)   = mae;
        summaryR2Pers(zi) = r2Pers;
        summaryOK(zi)    = true;

        %% 6. Console Metrics
        fprintf("Samples predicted: %d\n\n", size(XTest, 1));

        fprintf("  R²               : %8.4f\n", r2);
        fprintf("  RMSE             : %8.2f kWh\n", rmse);
        fprintf("  MAE              : %8.2f kWh\n", mae);
        if isfinite(mape)
            fprintf("  MAPE             : %8.2f %%\n", mape);
        end
        fprintf("\n  Persistence R²   : %8.4f\n", r2Pers);
        fprintf("  Persistence RMSE : %8.2f kWh\n", rmsePers);

        %% 7. Automated written analysis
        fprintf("\n--- Automated Performance Analysis ---\n");
        if r2 >= 0.80
            fprintf("Quality: EXCELLENT (R²=%.3f >= 0.80).\n", r2);
        elseif r2 >= 0.70
            fprintf("Quality: GOOD (R²=%.3f - publishable range).\n", r2);
        elseif r2 >= 0.50
            fprintf("Quality: ACCEPTABLE (R²=%.3f). Room for improvement.\n", r2);
        else
            fprintf("Quality: POOR (R²=%.3f < 0.50). Model fails to generalize.\n", r2);
        end

        deltaR2 = r2 - r2Pers;
        if deltaR2 > 0.05
            fprintf("Baseline: %s significantly outperforms persistence (+%.4f in R²).\n", m.modelType, deltaR2);
        elseif deltaR2 > 0
            fprintf("Baseline: %s marginally outperforms persistence (+%.4f).\n", m.modelType, deltaR2);
        else
            fprintf("Baseline: %s fails to outperform persistence (Delta=%.4f).\n", m.modelType, deltaR2);
        end

        orario        = hour(timeVectorTestMat);
        erroreAssoluto = abs(errors);
        errorePerOra  = zeros(24, 1);
        for h = 0:23
            mk = (orario == h);
            if any(mk)
                errorePerOra(h+1) = mean(erroreAssoluto(mk));
            end
        end
        [maxErr, oraMaxIdx] = max(errorePerOra);
        [minErr, oraMinIdx] = min(errorePerOra(errorePerOra > 0));
        oraMax = oraMaxIdx - 1;
        oraMin = find(errorePerOra == minErr, 1) - 1;

        fprintf("Performance by Hour:\n");
        fprintf("  Max Error: %02d:00 (MAE=%.1f kWh)\n", oraMax, maxErr);
        fprintf("  Min Error: %02d:00 (MAE=%.1f kWh)\n", oraMin, minErr);

        if oraMax >= 8 && oraMax <= 12
            fprintf("  Note: Morning peak error (8-12h) detected - common for commercial/office zones.\n");
        elseif oraMax >= 18 && oraMax <= 22
            fprintf("  Note: Evening peak error (18-22h) detected - common for residential zones.\n");
        end

        %% 8. Figure A — Scatter Plot Real vs Predicted
        figTitle = sprintf("Zone %d — %s — Real vs Predicted Scatter Plot", zId, m.modelType);
        figA = figure("Name", figTitle, "NumberTitle", "off", "Position", [50 500 520 480]);

        scatter(yTestReal, yPredReal, 30, [0.15 0.45 0.85], "filled", MarkerFaceAlpha=0.7);
        hold on;
        lims = [min([yTestReal; yPredReal]) * 0.95, max([yTestReal; yPredReal]) * 1.05];
        plot(lims, lims, "--r", LineWidth=1.8);
        hold off;

        xlabel("Real Consumption (kWh)", FontSize=11);
        ylabel("Predicted Consumption (kWh)", FontSize=11);
        title(sprintf("Zone %d — R²=%.3f | RMSE=%.1f kWh", zId, r2, rmse), FontSize=12);
        legend("Predictions", "y = x", Location="northwest");
        axis equal; axis tight; grid on;
        exportgraphics(figA, fullfile(reportDir, sprintf("Zone_%d_01_Scatter.png", zId)), Resolution=300);
        savefig(figA, fullfile(reportDir, "fig", sprintf("Zone_%d_01_Scatter.fig", zId)));

        %% 9. Figure B — Residuals in Time
        figTitle = sprintf("Zone %d — %s — Residuals Over Time", zId, m.modelType);
        figB = figure("Name", figTitle, "NumberTitle", "off", "Position", [590 500 700 320]);

        scatter(timeVectorTestMat, errors, 18, [0.2 0.5 0.9], "filled", MarkerFaceAlpha=0.75);
        yline(0, "--k", LineWidth=1);
        xlabel("Date", FontSize=11);
        ylabel("Error (kWh)", FontSize=11);
        title(sprintf("Zone %d — Test Set Residuals (MAE=%.1f kWh)", zId, mae), FontSize=12);
        grid on;
        exportgraphics(figB, fullfile(reportDir, sprintf("Zone_%d_02_Residuals.png", zId)), Resolution=300);
        savefig(figB, fullfile(reportDir, "fig", sprintf("Zone_%d_02_Residuals.fig", zId)));

        %% 10. Figure C — Error Breakdown by Hour
        figTitle = sprintf("Zone %d — %s — Hourly Error Breakdown", zId, m.modelType);
        figC = figure("Name", figTitle, "NumberTitle", "off", "Position", [50 50 800 350]);

        stdPerOra = zeros(24, 1);
        for h = 0:23
            mk = (orario == h);
            if any(mk); stdPerOra(h+1) = std(erroreAssoluto(mk)); end
        end

        ore = 0:23;
        bar(ore, errorePerOra, FaceColor=[0.95 0.6 0.2], EdgeColor="none", BarWidth=0.7);
        hold on;
        errorbar(ore, errorePerOra, stdPerOra / 2, stdPerOra / 2, "k.", LineWidth=1.2, CapSize=4);
        hold off;

        xlabel("Hour of Day (0–23)", FontSize=11);
        ylabel("Mean Absolute Error (kWh)", FontSize=11);
        title(sprintf("Zone %d — Average Error by Hour", zId), FontSize=12);
        xticks(0:23);
        xticklabels(string(0:23) + ":00");
        xtickangle(45);
        grid on; grid minor;
        exportgraphics(figC, fullfile(reportDir, sprintf("Zone_%d_03_HourlyError.png", zId)), Resolution=300);
        savefig(figC, fullfile(reportDir, "fig", sprintf("Zone_%d_03_HourlyError.fig", zId)));

        %% 11. Figure D — Time Series Comparison
        % Uses plotRegressionResults to handle discontinuous test segments properly
        figD = plotRegressionResults(yPredNorm, double(yTestNorm), normPs, cfg.target, timeVectorTestMat);
        set(figD, "Name", sprintf("Zone %d — Time Series Comparison", zId), ...
            "NumberTitle", "off", "Position", [590 80 800 350]);
        title(get(figD, "CurrentAxes"), sprintf("Zone %d — %s Prediction vs Real", zId, m.modelType), "FontSize", 12);
        exportgraphics(figD, fullfile(reportDir, sprintf("Zone_%d_04_TimeSeries.png", zId)), Resolution=300);
        savefig(figD, fullfile(reportDir, "fig", sprintf("Zone_%d_04_TimeSeries.fig", zId)));

    catch ex
        fprintf("WARNING - Zone %d: %s\n", zId, ex.message);
    end

    close all; % Close figures to save memory at the end of every zone
end

%% Final Cross-Zone Summary

fprintf("\n\n==============================================================\n");
fprintf("               PREDICTION CROSS-ZONE SUMMARY\n");
fprintf("                     Model: %s\n", netName);
fprintf("==============================================================\n");
fprintf("%-8s | %-8s | %-12s | %-12s | %-8s\n", "ZONE ID", "R²", "RMSE (kWh)", "MAE (kWh)", "Persist R²");
fprintf("%s\n", repmat('-', 1, 62));

for zi = 1:numel(zoneIds)
    if summaryOK(zi)
        fprintf("%-8d | %-8.4f | %-12.2f | %-12.2f | %-8.4f\n", ...
            summaryZone(zi), summaryR2(zi), summaryRMSE(zi), summaryMAE(zi), summaryR2Pers(zi));
    else
        fprintf("%-8d | %-8s | %-12s | %-12s | %-8s\n", ...
            summaryZone(zi), "FAIL", "FAIL", "FAIL", "FAIL");
    end
end
fprintf("%s\n", repmat('-', 1, 62));

validIdx = find(summaryOK);
if ~isempty(validIdx)
    [~, bestIdx] = max(summaryR2(validIdx));
    bestZone = summaryZone(validIdx(bestIdx));
    fprintf("Best R² achieved in: Zone %d (R²=%.4f)\n", bestZone, summaryR2(validIdx(bestIdx)));
end

diary off;
