## ⚡ Code Quality: Using Ruff

This project uses **Ruff** for all Python code quality enforcement.

Ruff is an extremely fast, all-in-one Python linter, formatter, and import sorter. It combines the functionality of multiple separate tools (like Flake8, Black, and isort) into a single, highly performant utility written in Rust.

### What Ruff Does

* **Linting:** Checks for errors, warnings, and potential bugs (e.g., unused imports, complexity, and common Python pitfalls).
* **Formatting:** Enforces consistent code style (e.g., line length of **88 characters**, double quotes) across the entire codebase.
* **Import Sorting:** Automatically organizes import statements according to the project's configuration.

### Where to Find Configuration

All Ruff settings, including enabled rule sets and file exclusions, are defined in the **[tool.ruff]** section of the [pyproject.toml](./pyproject.toml) file.

### How to Run

To quickly fix and format your code before committing, run these commands:

```bash
# Apply automatic linting fixes
ruff check . --fix

# Apply code formatting
ruff format .
