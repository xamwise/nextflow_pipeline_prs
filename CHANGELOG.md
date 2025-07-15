# nf-core/drugresponseeval: Changelog

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/)
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## v1.0dev - [date]

Initial release of nf-core/drugresponseeval, created with the [nf-core](https://nf-co.re/) template.

### `Added`

- Updated to the new template
- Added tests that run with docker, singularity, apptainer, and conda
- Added the docker container and the conda env.yml in the nextflow.config. We just need one container for all
  processes as this pipeline automates the PyPI package drevalpy.
- Added usage and output documentation.
- Added CurveCurator to preprocess curves of custom datasets

### `Fixed`

- Fixed linting issues
- Fixed bugs with path_data: can now be handled as absolute and relative paths

### `Dependencies`

### `Deprecated`
