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

## Continuous Learning

* Continuously add to AGENTS.md as you learn more about the codebase and its best practices.

<Environment Specific Instructions>
- Run tests through pytest
- You should run not only tests, but the code that you have changed.
- Run linters after changes, by doing 
```sh
lintrunner init
lintrunner -a
```

## Repository Information
* The main package lives in `torchLoom/` and includes modules like `weaver.py`,
  `monitor_cli.py`, and `weavelet.py`.
* The example training script is `examples/pytorch/mnist.py`.
* Start the NATS server with `./nats/nats-server -c ./nats/nats.conf` and run
  the weaver using `python -m torchLoom.weaver`.
* The interactive CLI can be launched with `python -m torchLoom.monitor_cli`.
* To test training end to end, run the training script after the server and
  weaver are running.
* Create the Python environment with `conda env create -f environment.yaml` and
  activate it before running any commands.
* All tests are under the `tests/` directory and should be run with `pytest`.
* Additional documentation is in the `docs/` folder with design notes in
  `docs/design`.
