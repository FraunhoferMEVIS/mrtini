# MRTini GitLab unit-test pipeline

Extract this package into the repository root, alongside src/ and your existing
tests/. It adds only .gitlab-ci.yml at the root; all CI support lives in tests/ci/.
If the root file is still the demonstration template, replace it. If it contains
real jobs, merge the configuration deliberately: this included file defines
workflow rules and the stages list.

## Activate

1. Commit .gitlab-ci.yml and tests/ci/ together with the existing test suite.
2. Push to GitLab. Open Build > Pipelines to see the job and its logs.
3. Open a merge request (GitLab's name for a pull request). Subsequent pushes
   run a merge-request pipeline, avoiding a duplicate branch pipeline.
4. In Settings > Merge requests > Merge checks, enable Pipelines must succeed
   if you want failing tests or coverage below 90% to block merging.
5. Run a pipeline on the default branch too, so merge-request reports have
   baseline test and coverage results for comparison.

Branch pushes, tag pushes, merge-request events, and manually created pipelines
from the GitLab UI are supported. Schedules and API-triggered pipelines are not
enabled by these workflow rules.

## Runner requirements

The project needs an enabled Linux x86_64 GitLab Runner with a Docker or
Kubernetes executor, able to pull python:3.10-slim-bookworm and download wheels
from PyPI and download.pytorch.org. No Docker-in-Docker or privileged job is
needed. On GitLab.com, enable an available hosted runner; on a self-managed
instance, ask your administrator for a runner.

This job has no runner tags, so it needs a runner accepting untagged jobs.
If your organization's runner requires tags, add its actual tags to the job.
A job stuck pending usually indicates runner availability, tags, protection,
or quota settings rather than a pytest failure. A shell executor ignores image:
and would require its own correctly provisioned Python environment.

## What GitLab shows

- Job/pipeline passed or failed, using the test runner's actual exit status.
- Individual test results through the JUnit report in pipeline Tests and the
  merge-request test summary.
- The real combined statement/branch coverage percentage from pytest-cov's
  TOTAL line. The 90% threshold is enforced by pytest-cov, not by the regex.
- Changed-line coverage annotations from Cobertura XML after report processing.
  These annotations describe line coverage; they do not themselves enforce the
  combined statement/branch threshold.
- Downloadable HTML/XML/JSON reports in job artifacts, retained for 30 days.
  Download artifacts and open tests/reports/mrtini/html/index.html locally.

Reports are uploaded after test failures too, when the test process generated
them. Installation failures cannot produce test reports. Artifacts may be visible
to people with project access; keep ordinary GitLab project permissions in mind.

## Scope and dependencies

This runs your existing independent Python unit suite, including SO3 and spinor
tests. The suite's existing fixtures isolate native, parser, and UI boundaries;
the Rust implementation and real parser/service integration are not exercised.
It does not run a GPU integration test, launch reconstruction services, or deploy
anything. Those can be separate jobs later.

The environment follows the dev container's Python 3.10 and PyTorch 2.8 versions,
using CPU PyTorch and a headless matplotlib backend. It intentionally does not
execute post_create.sh, whose setup builds Rust and installs GUI/reconstruction
dependencies. No source, tests, or coverage exclusions are changed.

Top-level numerical/test dependencies are pinned; this is not a complete
transitive lockfile. The first real GitLab run validates this clean Python 3.10
environment against your current release repository. Local checks validated the
YAML, report paths, coverage regex, and pipeline-rule cases; a GitLab runner and
fresh container were not available for an end-to-end CI run here.

Optional README badges (replace the host, namespace, repository, and branch):

```markdown
[![pipeline](https://gitlab.com/NAMESPACE/PROJECT/badges/main/pipeline.svg)](https://gitlab.com/NAMESPACE/PROJECT/-/pipelines)
[![coverage](https://gitlab.com/NAMESPACE/PROJECT/badges/main/coverage.svg?job=mrtini-unit-tests)](https://gitlab.com/NAMESPACE/PROJECT/-/pipelines)
```

Official references:
- https://docs.gitlab.com/ci/testing/unit_test_reports/
- https://docs.gitlab.com/ci/testing/code_coverage/
- https://docs.gitlab.com/ci/yaml/workflow/
- https://docs.gitlab.com/user/project/merge_requests/auto_merge/#require-a-successful-pipeline-for-merge
