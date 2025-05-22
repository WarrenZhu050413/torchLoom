# Instructions

## General Guidelines
 * Plan First: Before coding, create a detailed implementation plan in plan.md, explaining actions and rationale. Update this plan as you complete tasks.
 * Prioritize Simplicity: Favor simple, error-free approaches over complex ones, unless specifically requested. As simple as possible, but no simpler.
 * Review and Refactor: After implementing a feature, review your code for correctness, modularity, and potential code reuse.
 * Don't Alter Tests to Pass: If tests fail, fix the code, not the tests (unless the tests themselves are flawed).
 * Explain Design Decisions: Clearly articulate your design choices. If fixing bugs, explain their cause to prevent future occurrences.

## Design Guide
 * Simple Tests: Keep tests as simple as possible, akin to unit tests.
 * Code Consistency: Reuse existing code and maintain the established coding style.
 * Test without Changing Code: When writing tests, avoid modifying existing code unless it contains bugs.

## Style Guide
 * No Numbered Comments: Do not use numbered lists (e.g., 1., 2., 3...) in comments.

## Details
 * NVIDIA Environment: When setting up NVIDIA environments, verify compatibility using nvidia-smi and checking the installed version.
 * Conda Usage: Always initialize (conda init) and activate the correct conda environment before running commands.
 * Linter Settings: Do not modify linter configurations.
* Remember to import all the libraries that you are using. Also, don't reimport the same library twice.

## Testing
* After you implement any change, always run the tests to ensure that you didn't break anything.
* If the feature you implement is not covered by the tests, add tests to cover it.

<Environment Specific Instructions>
- Run tests through pytest