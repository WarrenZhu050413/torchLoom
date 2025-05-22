# Contributing to torchLoom

Notes: This contributor guide is adapted from torchft's contributor guide.

We want to make contributing to this project as easy and transparent as possible.

## TL;DR

We appreciate all contributions. If you are interested in contributing to torchft, there are many ways to help out.
Your contributions may fall into the following categories:

- It helps the project if you can

  - Report issues that you're facing
  - Give a :+1: on issues that others reported and that are relevant to you

- Answering questions on the issue tracker, investigating bugs are very valuable contributions to the project.

- You would like to improve the documentation. This is no less important than improving the library itself! If you find
  a typo in the documentation, do not hesitate to submit a GitHub pull request.

- If you would like to fix a bug:

  - comment on the issue that you want to work on this issue
  - send a PR with your fix, see below.

- If you plan to contribute new features, utility functions or extensions, please first open an issue and discuss the
  feature with us.
- If you would like to feature a usage example in our documentation, discuss that with us in an issue.

## Issues

We use GitHub issues to track public bugs. Please follow the existing templates if possible and ensure that the
description is clear and has sufficient instructions to be able to reproduce the issue.

## Development installation

torchLoom is written in Python and uses NATS as its messaging system.

You can set up by running the following commands:

```sh
chmod +x setup.sh
./setup.sh
```

## Pull Requests

We actively welcome your pull requests.

1. Fork the repo and create your branch from `main`.
2. If you've added code that should be tested, add tests.
3. If you've changed APIs, update the documentation and examples.
4. Ensure the test suite passes.
5. If you haven't already, complete the Contributor License Agreement ("CLA").

### Code style

`torchLoom` enforces a fairly strict code format with tools such as black.

```shell
pip install lintrunner lintrunner-adapters
lintrunner init
lintrunner -a
```

### Tests

We use `pytest` as our testing framework. To execute a specific test, use the following command:

```sh
pytest torchLoom/loom_test.py loom_test.py
```

To run the entire suite of tests:

```sh
$ scripts/test.sh
```

## License

By contributing to torchft, you agree that your contributions will be licensed under the LICENSE file in the root
directory of this source tree.
