# Rule: Mandatory Step-by-Step Reporting in docs/report.md

## Trigger
Apply immediately whenever an implementation step (Steps 1 through 7) or a major milestone is completed, and before declaring the step finished or advancing to the next step.

## Mandatory Requirement
You MUST update `docs/report.md` with the empirical findings, quantitative results, and core conclusions produced during that step.

## Guidelines & Constraints
1. **Style & Tone:**
   - Keep entries strictly objective, concise, and technical.
   - Focus directly on "What was done", "Key Takeaways", and "Empirical Conclusions / Lessons Learned".
   - Follow the exact format and density established in the Step 1 section of `docs/report.md`.

2. **Strict Prohibitions:**
   - **NO Executive Summaries:** Do not write narrative executive summaries, high-level corporate abstracts, or redundant intros.
   - **NO Decorative Fluff:** Avoid verbose padding, promotional language, or unnecessary prose.
   - **NO Overkill Sub-numbering:** Do NOT use deep decimal section hierarchies (e.g., avoid sections like `2.5`, `1.1.3`). Use clean Markdown headers (`## Step X`, `### Topic`).

3. **Required Content per Step:**
   - **Key Metrics / Counts:** Exact numbers (e.g., accuracy, coverage %, review queue reduction %, parse success rates).
   - **Architecture / Methodological Decisions:** Brief bullets on algorithms, formulas, or key file roles.
   - **Empirical Cases / Surprises:** Concrete examples or edge cases discovered in the data that validate or challenge our hypotheses (e.g., formatting slips, legacy label flaws).

4. **Workflow Checkpoint:**
   - Updating `docs/report.md` is a hard completion criterion. Never mark a step as completed in `docs/project_guide.md` without first documenting its conclusions in `docs/report.md`.
