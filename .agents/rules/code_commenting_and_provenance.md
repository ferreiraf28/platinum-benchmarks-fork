# Rule: Code Commenting and Provenance Documentation

## Trigger
Applies unconditionally whenever creating, editing, or refactoring any file (code, tests, scripts, or configuration) in this repository.

## Mandatory Requirements

### 1. File-Level Provenance & Purpose Header
At the very top of every edited or newly created file, include an objective header docstring/comment specifying:
1. **Provenance Category (Must be one of the three):**
   - `[ORIGINAL - UNMODIFIED]`: Original file from the upstream MIT repository with zero modifications.
   - `[ORIGINAL - MODIFIED]`: Original file from the upstream MIT repository with specific changes (must concisely list the modifications, e.g., *"Added package import fallback for local test execution"*).
   - `[FORK NEW]`: New file created specifically for this project fork.
2. **File Purpose & Functioning:**
   - 1-3 concise sentences explaining the role of the file within the pipeline, what problem it solves, and how it interacts with other components.

### 2. Function, Method & Inline Commenting Standards
- **Technical Focus:** Comments and docstrings must explain *what* the code does, *why* particular logic or thresholds were chosen, and *how* inputs map to outputs.
- **Formulas & Edge Cases:** State mathematical formulas or edge-case handling clearly (e.g., handling ties, zero division, empty text).
- **Concise & Didactic:** Keep explanations clear and directly to the point so any researcher can immediately understand the mechanics.

### 3. Strict Prohibitions
- **NO Personal Attributions:** Never include author names, signatures, email addresses, or timestamps (`@author`, `Created by...`).
- **NO Conversational Filler:** Avoid informal remarks, placeholder comments (`TODO later`), or speculative commentary. Focus strictly on functioning and purpose.

### 4. Language
- All comments, docstrings, and headers MUST be written in English.
