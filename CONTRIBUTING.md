# Contributing Guidelines

## Pull Request Checklist

Before sending your pull requests, make sure you followed this list.

- Read [contributing guidelines](CONTRIBUTING.md).
- Read the [Contributor License Agreement (CLA)](CONTRIBUTING.md#Contributor-License-Agreement).
- Check if my changes are consistent with the [guidelines](CONTRIBUTING.md#Contribution-guidelines-and-standards).
- Changes are consistent with the [coding style](CONTRIBUTING.md#Python-coding-style).
- Commits are consistent with the [commit conventions](CONTRIBUTING.md#Commit-conventions).

## How to become a contributor and submit your own code

### Contributor License Agreement

By contributing to this codebase You understand and agree that this project and contributions to it are public and that a record of the contribution (including all personal information You submit with it, including Your full name and email address) is maintained indefinitely and may be redistributed consistent with this project, compliance with the open source license(s) involved, and maintenance of authorship attribution.

You accept and agree to the following terms and conditions for Your present and future Contributions submitted to Beacon Platform Inc. Except for the license granted herein to Beacon Platform Inc and recipients of software distributed by Beacon Platform Inc, You reserve all right, title, and interest in and to Your Contributions.

1. Definitions.

    "You" (or "Your") shall mean the copyright owner or legal entity authorized by the copyright owner that is making this Agreement with Beacon Platform Inc. For legal entities, the entity making a Contribution and all other entities that control, are controlled by, or are under common control with that entity are considered to be a single Contributor. For the purposes of this definition, "control" means (i) the power, direct or indirect, to cause the direction or management of such entity, whether by contract or otherwise, or (ii) ownership of fifty percent (50%) or more of the outstanding shares, or (iii) beneficial ownership of such entity.

    "Contribution" shall mean any original work of authorship, including any modifications or additions to an existing work, that is intentionally submitted by You to Beacon Platform Inc for inclusion in, or documentation of, any of the products owned or managed by Beacon Platform Inc (the "Work"). For the purposes of this definition, "submitted" means any form of electronic, verbal, or written communication sent to Beacon Platform Inc or its representatives, including but not limited to communication on electronic mailing lists, source code control systems, and issue tracking systems that are managed by, or on behalf of, Beacon Platform Inc for the purpose of discussing and improving the Work, but excluding communication that is conspicuously marked or otherwise designated in writing by You as "Not a Contribution."

2. Grant of Copyright License. Subject to the terms and conditions of this Agreement, You hereby grant to Beacon Platform Inc and to recipients of software distributed by Beacon Platform Inc a perpetual, worldwide, non-exclusive, no-charge, royalty-free, irrevocable copyright license to reproduce, prepare derivative works of, publicly display, publicly perform, sublicense, and distribute Your Contributions and such derivative works.

3. Grant of Patent License. Subject to the terms and conditions of this Agreement, You hereby grant to Beacon Platform Inc and to recipients of software distributed by Beacon Platform Inc a perpetual, worldwide, non-exclusive, no-charge, royalty-free, irrevocable (except as stated in this section) patent license to make, have made, use, offer to sell, sell, import, and otherwise transfer the Work, where such license applies only to those patent claims licensable by You that are necessarily infringed by Your Contribution(s) alone or by combination of Your Contribution(s) with the Work to which such Contribution(s) was submitted. If any entity institutes patent litigation against You or any other entity (including a cross-claim or counterclaim in a lawsuit) alleging that your Contribution, or the Work to which you have contributed, constitutes direct or contributory patent infringement, then any patent licenses granted to that entity under this Agreement for that Contribution or Work shall terminate as of the date such litigation is filed.

4. You represent that you are legally entitled to grant the above license. If your employer(s) has rights to intellectual property that you create that includes your Contributions, you represent that you have received permission to make Contributions on behalf of that employer, that your employer has waived such rights for your Contributions to Beacon Platform Inc, or that your employer has executed a separate Corporate CLA with Beacon Platform Inc.

5. You represent that each of Your Contributions is Your original creation (see section 7 for submissions on behalf of others). You represent that Your Contribution submissions include complete details of any third-party license or other restriction (including, but not limited to, related patents and trademarks) of which you are personally aware and which are associated with any part of Your Contributions.

6. You are not expected to provide support for Your Contributions, except to the extent You desire to provide support. You may provide support for free, for a fee, or not at all. Unless required by applicable law or agreed to in writing, You provide Your Contributions on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied, including, without limitation, any warranties or conditions of TITLE, NON- INFRINGEMENT, MERCHANTABILITY, or FITNESS FOR A PARTICULAR PURPOSE.

7. Should You wish to submit work that is not Your original creation, You may submit it to Beacon Platform Inc separately from any Contribution, identifying the complete details of its source and of any license or other restriction (including, but not limited to, related patents, trademarks, and license agreements) of which you are personally aware, and conspicuously marking the work as "Submitted on behalf of a third-party: [named here]".

8. You agree to notify Beacon Platform Inc of any facts or circumstances of which you become aware that would make these representations inaccurate in any respect.

### Contributing code

If you have improvements, send us your pull requests! For those
just getting started, Github has a
[how to](https://help.github.com/articles/using-pull-requests/).

Beacon Platform Inc team members will be assigned to review your pull requests. Once the
pull requests are approved and pass continuous integration checks, a Beacon Platform Inc
team member will apply `ready to pull` label to your change. This means we are
working on getting your pull request submitted to the repository.

If you want to contribute, start working through the codebase,
navigate to the Github "issues" tab and start
looking through interesting issues. If you are not sure of where to start, then
start by trying one of the smaller/easier issues here i.e.
issues with the "good first issue" label
and then take a look at the
issues with the "contributions welcome" label.
These are issues that we believe are particularly well suited for outside
contributions, often because we probably won't get to them right now. If you
decide to start on an issue, leave a comment so that other people know that
you're working on it. If you want to help out, but not alone, use the issue
comment thread to coordinate. If somebody is already assigned to an issue you
would like to tackle, do not start work on it without confirming that you will
not be duplicating efforts by doing so.

### Contribution guidelines and standards

Before sending your pull request for review, make sure your changes are consistent
with the guidelines and follow the coding style.

#### General guidelines and philosophy for contribution

*   As this project is in its infancy, improvements around all aspects of
    the APIs and structure as well as the addition of models, tools, tests,
    examples, and documentation are greatly appreciated.
*   Include unit tests when you contribute new features, as they help to a)
    prove that your code works correctly, and b) guard against future breaking
    changes to lower the maintenance cost.
*   Bug fixes also generally require unit tests, because the presence of bugs
    usually indicates insufficient test coverage.
*   Keep API compatibility in mind when you change code. Reviewers of your pull
    request will comment on any API compatibility issues.
*   When you contribute a new feature, the maintenance burden is (by default)
    transferred to the Beacon Platform Inc team. This means that the benefit
    of the contribution must be compared against the cost of maintaining the
    feature.

#### License

Include a license at the top of new files.

**Python license**

    # Copyright (C) 2020 Beacon Platform Inc. - All Rights Reserved.
    # License: MIT

#### Python coding style

Changes to the Python code should conform to the
[Google Python Style Guide](https://github.com/google/styleguide/blob/gh-pages/pyguide.md).

Use `pylint` to check your Python changes. To install `pylint` and check a file
against the style definition:

```bash
pip install pylint
pylint myfile.py
```

Note `pylint` should run from the top level project directory.

#### Commit conventions

Commits to the repository must follow the Conventional Commits v1.0.0 standard.

- All commits must use an appropriate type and description in the imperative tense.
- Squash commits together that contain unfinished work or contribute to the same change.

## Attribution

These Contributing Guidelines are adapted from the [TensorFlow Contributing Guidelines](https://github.com/tensorflow/tensorflow/blob/master/CONTRIBUTING.md) and associated CLAs.
