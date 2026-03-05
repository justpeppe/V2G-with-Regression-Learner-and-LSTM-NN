# MATLAB Coding Standards Summary

## 1. Naming Guidelines
- **General Principles**: Use English, be descriptive, and avoid abbreviations. Prevent "shadowing" by avoiding names of existing MATLAB functions.
- **Length**: Keep variable and function names <= 32 characters.
- **Variables**: Use `lowerCamelCase`. Use a suffix for pluralization (e.g., `pointArray`). Short names are only for math symbols, loop iterators, or temporary variables.
- **Functions/Methods**: Use `lowerCamelCase` or lowercase. Names should be verb phrases (e.g., `calculateResult`) or nouns if creating an object. Use "2" for conversions (e.g., `struct2table`) and "is"/"has" for logical outputs.
- **Classes**: Use `UpperCamelCase` (if inside a namespace) and nouns for objects (e.g., `PrintServer`) or adjectives for mixins (e.g., `Copyable`).
- **Properties/Events**: Use `UpperCamelCase`. Properties should be nouns; events should describe the action.

## 2. Statements and Expressions
- **Style**: One statement per line. Use variables instead of hard-coded literals. Write floating points with a leading digit (e.g., `0.5`).
- **Data Management**: Avoid global variables; minimize persistent variables. Define all struct fields in a single block. Use **string arrays** instead of cell arrays for text.
- **Operations**: Use `fileparts`, `fullfile`, and `filesep` for platform-independent paths. Use parentheses to clarify logic. **Never** use `==` or `~=` for floating-point comparisons.
- **Control Flow**: 
    - Limit nesting to 5 levels.
    - **Pre-allocate** arrays before loops.
    - Do not modify iterators inside `for` loops.
    - `switch` statements must include an `otherwise` block.
- **Function Calls**: Use empty parentheses `()` for zero-argument calls. Use `~` to ignore outputs and `Name=Value` syntax for arguments.

## 3. Formatting
- **Spacing**: Use **4 spaces** for indentation (no tabs). Put spaces after commas/semicolons and on both sides of assignment, relational, and logical operators.
- **Compactness**: No spaces around the colon (`:`), multiply/divide/exponent operators, or inside brackets/parentheses.
- **Lines**: Max length of **120 characters**. Use single blank lines to separate logical sections.

## 4. Code Comments
- **Standard**: Use English and a space after `%`. Use `%%` for sections.
- **Documentation**: Place the **H1 line** (brief description) immediately after the function declaration. Follow with help text covering syntax, inputs, and outputs.
- **Placement**: Align comments with the code they explain; place them just before the code block.

## 5. Function Authoring
- **Structure**: Filename must match the top-level function. Always use the `end` keyword. Keep exclusive functions as local functions in the same file.
- **Arguments**: Limit to **6 inputs** and **4 outputs**. Use the `arguments` block for validation. favor name-value pairs for optional data.
- **Optimization**: Element-wise functions should handle any array shape (return same shape as input). Avoid `nested` functions when `local` functions suffice.

## 6. Class Authoring
- **Design**: Filename must match class name. Prefer **value classes**; use **handle classes** only for objects with identity/state (e.g., hardware, UI).
- **Attributes**: Use `Sealed` unless the class is designed for inheritance.
- **Encapsulation**: Make properties/methods as restrictive as possible (Private/Protected). Use **property validation syntax** instead of custom set-methods. Use `Dependent` properties only when computed from other data.

## 7. Error Handling
- **Quality**: Fix all Code Analyzer warnings before submission.
- **Messages**: State the problem, the solution, or both. Be specific.
- **Stability**: Use `try-catch` and `onCleanup` to reset global states or settings after an error. Use `MException` to identify specific errors in catch blocks. Avoid `throwAsCaller`.

## 8. Modern MATLAB Features
- **Strings**: Always use double quotes `"` for text.
- **Access**: Use dot notation `obj.PropertyName` rather than `get`/`set` methods.
- **Syntax**: Use `Name=Value` for arguments (R2021a+) and `arguments` blocks (R2019b+) for robust validation.
