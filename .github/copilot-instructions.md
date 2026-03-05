---
applyTo: "**"
---
# GitHub Copilot Instructions — MATLAB Project

## MCP Tools: When and How to Use Them

You have access to the MATLAB MCP Core Server. These tools execute real MATLAB
in the local installation. Always use them — never simulate or guess MATLAB output.

### Mandatory workflow before any code response:

1. **`detect_matlab_toolboxes`** — Run ONCE at the start of every conversation.
   Use only functions from toolboxes that are actually available.
   Reason: suggesting Signal Processing Toolbox functions on a base-only install
   causes silent failures that are hard to debug.

2. **`check_matlab_code`** — Run on every code block BEFORE showing it to the user.
   Fix all warnings, not just errors.
   Reason: MATLAB warnings often indicate real bugs (e.g. implicit expansion issues
   in older releases, shadowed built-ins).

3. **`evaluate_matlab_code`** — Run to verify output, especially for:
   - Numerical results (floating point, array shapes)
   - Figure generation
   - Any code with loops or recursion
   Reason: MATLAB indexing starts at 1 and has non-obvious broadcasting rules —
   verification prevents off-by-one errors that look correct in review.

4. **`run_matlab_file`** — Use for multi-function files or scripts that require
   the full workspace context (e.g. scripts that call `load`, `readtable`, etc.).

5. **`run_matlab_test_file`** — Run after generating or modifying test files.
   Show the full test report, including passed tests — not just failures.
   Reason: test output confirms the framework loaded correctly and all cases ran.

### Tool decision table

| Situation | Tool to use |
|---|---|
| New conversation starts | `detect_matlab_toolboxes` |
| Writing any function or script | `check_matlab_code` → `evaluate_matlab_code` |
| Script needs workspace variables | `run_matlab_file` |
| After writing or editing tests | `run_matlab_test_file` |
| Checking if a toolbox function exists | `detect_matlab_toolboxes` |

---

## MATLAB Code Standards

### Input Validation — Always use `arguments` block

```matlab
% CORRECT — validates before execution, gives clear error messages
function result = computeRMS(signal, windowSize)
    arguments
        signal    (1,:) double {mustBeReal, mustBeFinite}
        windowSize (1,1) double {mustBePositive, mustBeInteger}
    end
    result = sqrt(movmean(signal.^2, windowSize));
end

% WRONG — error message points to internal code, not the caller
function result = computeRMS(signal, windowSize)
    result = sqrt(movmean(signal.^2, windowSize));
end
```
Reason: `arguments` block errors include the caller's variable name in the message,
dramatically faster to debug than "Index exceeds matrix dimensions" at line 47.

### Error IDs — Always structured

```matlab
% CORRECT
error('packageName:functionName:invalidInput', ...
      'windowSize must be a positive integer, got %d', windowSize);

% WRONG
error('Invalid input');
```
Reason: structured IDs allow `verifyError` in unit tests to target specific errors.
Without them, tests cannot distinguish between expected and unexpected errors.

### Array Preallocation — Always before loops

```matlab
% CORRECT
n = 1000;
results = zeros(1, n);
for i = 1:n
    results(i) = expensiveComputation(i);
end

% WRONG — MATLAB re-allocates on every iteration, 10-100x slower
results = [];
for i = 1:n
    results(end+1) = expensiveComputation(i);
end
```
Reason: dynamic growth triggers O(n²) memory copies. Always preallocate,
even in non-performance-critical code — it signals intent and prevents bugs
when the loop body conditionally skips iterations.

### Vectorization — Prefer over loops for array operations

```matlab
% CORRECT
normalizedData = (data - mean(data)) / std(data);

% WRONG for large arrays — same result, 50-200x slower
for i = 1:length(data)
    normalizedData(i) = (data(i) - mean(data)) / std(data);
end
```
Exception: use loops when iteration order matters or when each step depends
on the previous result (e.g. IIR filters, recursive algorithms).

### Naming Conventions

| Element | Convention | Example |
|---|---|---|
| Functions | camelCase | `processSignal`, `loadDataset` |
| Variables | camelCase | `inputMatrix`, `filterOrder` |
| Constants | UPPER_SNAKE_CASE | `MAX_ITERATIONS`, `SAMPLE_RATE` |
| Classes | PascalCase | `SignalProcessor`, `DataLoader` |
| Test classes | PascalCase + `Test` suffix | `SignalProcessorTest` |
| Private methods | underscore prefix | `_computeInternal` |

### Help Blocks — Required for all public functions

```matlab
function output = processSignal(signal, fs, cutoffHz)
%PROCESSSIGNAL  Apply low-pass filter to input signal.
%
%   OUTPUT = PROCESSSIGNAL(SIGNAL, FS, CUTOFFHZ) filters SIGNAL sampled
%   at FS Hz using a 4th-order Butterworth low-pass filter with cutoff
%   frequency CUTOFFHZ Hz.
%
%   Inputs:
%     SIGNAL    - (1,N) double, input time-domain signal
%     FS        - (1,1) double, sampling frequency in Hz (must be > 0)
%     CUTOFFHZ  - (1,1) double, cutoff frequency in Hz (must be < FS/2)
%
%   Output:
%     OUTPUT    - (1,N) double, filtered signal, same size as SIGNAL
%
%   Example:
%     fs = 1000;
%     t = 0:1/fs:1;
%     noisy = sin(2*pi*50*t) + 0.5*randn(size(t));
%     clean = processSignal(noisy, fs, 100);
%
%   See also: BUTTER, FILTFILT, BANDPASSFILTER

    arguments
        signal    (1,:) double {mustBeReal, mustBeFinite}
        fs        (1,1) double {mustBePositive}
        cutoffHz  (1,1) double {mustBePositive}
    end
    % ... implementation
end
```
Reason: help blocks are the only documentation users see when they type
`help processSignal` or hover in the Editor. They also serve as the spec
when writing unit tests.

---

## Unit Testing Standards

Use `matlab.unittest.TestCase`. Every public function must have a corresponding
`*Test.m` file in the `tests/` folder.

### Required test categories per function

```matlab
classdef ProcessSignalTest < matlab.unittest.TestCase

    methods (Test)
        % 1. Nominal case — typical input, verify output shape and values
        function testNominalFiltering(tc)
            fs = 1000; t = 0:1/fs:0.1;
            signal = sin(2*pi*50*t);
            output = processSignal(signal, fs, 200);
            tc.verifySize(output, size(signal));
            tc.verifyLessThanOrEqual(max(abs(output)), max(abs(signal)) + 1e-9);
        end

        % 2. Edge case — boundary inputs
        function testSingleSample(tc)
            output = processSignal(1.0, 1000, 100);
            tc.verifySize(output, );
        end

        % 3. Expected error — wrong input triggers correct error ID
        function testNegativeSamplingRate(tc)
            tc.verifyError(@() processSignal([1 2 3], -1, 100), ...
                'processSignal:invalidFs');
        end

        % 4. Numerical precision — floating point comparisons with tolerance
        function testKnownOutputPrecision(tc)
            % DC signal should pass through unchanged
            signal = ones(1, 100);
            output = processSignal(signal, 1000, 100);
            tc.verifyEqual(output, signal, 'AbsTol', 1e-9);
        end
    end
end
```

After generating tests, always run them with `run_matlab_test_file` and
show the full result table.

---

## Response Behavior

- If a toolbox required by the user's request is not available
  (detected by `detect_matlab_toolboxes`), immediately say so and offer
  a base-MATLAB alternative.

- If `check_matlab_code` returns warnings, fix them silently and note
  what was changed at the bottom of the response.

- If `evaluate_matlab_code` produces a different result than expected,
  investigate before showing the user — do not show code that fails
  verification.

- Avoid unnecessary apologies or hedging. State what the code does
  and why it is correct, referencing the tool output as evidence.

- For numerical results, always include the unit (Hz, seconds, dB, etc.)
  in variable names or comments — not just in the help block.
