# Repository Reorganization Proposal

## 1. Introduction

This document outlines a proposal for reorganizing the repository to improve clarity, maintainability, and ease of use for both new and existing contributors. The proposed changes are based on an analysis of the current file structure and aim to address several identified issues, including ambiguous naming, overlapping responsibilities, and a monolithic `library` directory.

## 2. Key Issues Identified

- **Overly Broad `library` Directory**: The `library` directory currently contains a mix of core model definitions, training utilities, and third-party code (`vendor`). This makes it difficult to distinguish between the core, reusable components and the more application-specific training logic.
- **Ambiguous `tools` Directory**: The `tools` directory contains a variety of scripts, from data preprocessing to model management. The name "tools" is too generic and doesn't accurately reflect the purpose of these scripts.
- **Unclear Entrypoints**: The `training` directory contains the main training scripts, but its name doesn't immediately convey that these are the primary entrypoints for running the code.
- **Redundant Naming**: Some files have redundant or verbose names, such as `sdxl_train.py` and `sdxl_train_network.py`, which could be simplified for better readability.
- **Third-party Code**: The presence of a `vendor` directory for third-party libraries is not a standard practice and can lead to issues with dependency management. It is better to manage dependencies through `requirements.txt`.

## 3. Proposed Directory Structure

To address these issues, the following new directory structure is proposed:

```
.
├── src/
│   ├── a1111/
│   │   ├── api.py
│   │   ├── constants.py
│   │   ├── util.py
│   ├── accelerate_util/
│   │   ├── autodevice.py
│   │   ├── easy_accelerator.py
│   │   ├── fix_optimizer.py
│   ├── data_util/
│   │   ├── image_util.py
│   │   ├── dataset.py
│   │   ├── bucketing.py
│   │   ├── caching.py
│   ├── model_util/
│   │   ├── ...
│   ├── network_util/
│   │   ├── ...
│   ├── optim_util/
│   │   ├── ...
│   ├── train_util/
│   │   ├── ...
│   ├── ...
├── scripts/
│   ├── data_processing/
│   │   ├── ...
│   ├── model_management/
│   │   ├── ...
│   ├── upscaling/
│   │   ├── ...
│   ├── visualization/
│   │   ├── ...
├── training/
│   ├── train_sdxl.py
│   ├── train_network.py
│   ├── ...
├── docs/
│   ├── ...
├── tests/
│   ├── ...
├── .gitignore
├── LICENSE.md
├── README.md
├── requirements.txt
├── setup.py
```

## 4. Rationale for Changes

### 4.1. Introduce a `src` Directory

- **Current State**: The `library` directory is a catch-all for various types of code.
- **Proposed Change**: Rename `library` to `src` (source) to better reflect its role as the primary location for the project's source code. This is a common convention in Python projects and makes the structure more intuitive.
- **Subdirectories**: Within `src`, the existing subdirectories will be renamed to follow a more consistent `_util` suffix convention, clearly indicating that they are utility modules. For example:
  - `library/train` -> `src/train_util`
  - `library/models` -> `src/model_util`
  - `library/networks` -> `src/network_util`
  - and so on.

### 4.2. Rebrand `tools` as `scripts`

- **Current State**: The `tools` directory contains scripts for various tasks.
- **Proposed Change**: Rename `tools` to `scripts`. This is a more descriptive name for a directory containing standalone scripts that are not part of the main library but are used for project-related tasks.

### 4.3. Simplify Training Scripts

- **Current State**: The `training` directory contains scripts with long, descriptive names (e.g., `sdxl_train.py`).
- **Proposed Change**: The `training` directory will remain, but the scripts within it can be simplified. For instance, `sdxl_train.py` can be renamed to `train_sdxl.py` for better consistency. The purpose of this directory is to house the main entrypoints for training models.

### 4.4. Eliminate the `vendor` Directory

- **Current State**: The `vendor` directory contains third-party libraries.
- **Proposed Change**: The `vendor` directory should be removed. Dependencies should be managed exclusively through the `requirements.txt` file. This ensures that dependencies are installed in a standard way and makes it easier to track and update them.

## 5. File and Folder Mapping

The following table maps the current file and folder locations to their proposed new locations:

| Current Path                       | Proposed New Path                   | Notes                               |
| ---------------------------------- | ----------------------------------- | ----------------------------------- |
| `library/`                         | `src/`                              | Renamed for clarity.                |
| `library/models/`                    | `src/model_util/`                   | Consistent naming convention.       |
| `library/networks/`                  | `src/network_util/`                 | Consistent naming convention.       |
| `library/optimizations/`             | `src/optim_util/`                   | Consistent naming convention.       |
| `library/train/`                     | `src/train_util/`                   | Consistent naming convention.       |
| `library/utils/`                     | `src/`                              | Flattened into `src` directly.      |
| `library/vendor/`                    | (removed)                           | Dependencies in `requirements.txt`. |
| `tools/`                           | `scripts/`                          | Renamed for clarity.                |
| `training/sdxl_train.py`             | `training/train_sdxl.py`            | Simplified and consistent naming.   |
| `training/sdxl_train_network.py`     | `training/train_network.py`         | Simplified and consistent naming.   |

## 6. Next Steps

This proposal is intended to be a guide for the reorganization of the repository. The next steps would be to:

1. **Review and Approve**: Discuss and approve the proposed changes.
2. **Implement**: Execute the file and folder renames and moves as outlined in this document.
3. **Update Imports**: Update all import statements in the codebase to reflect the new structure.
4. **Test**: Thoroughly test the codebase to ensure that all functionality remains intact.
5. **Update Documentation**: Update the `README.md` and any other relevant documentation to reflect the new structure.

By following this plan, we can achieve a more organized and maintainable codebase that is easier to navigate and contribute to.
