# Welcome to the contributing guide <!-- omit in toc -->

Thank you for investing your time in contributing to our project! :sparkles:.

Read our [Code of Conduct](./CODE_OF_CONDUCT.md) to keep our community approachable and respectable.

## New contributor guide

**Start here:** Visit [README](./README.md) for installation and development guidelines.

## Issues

**If you find a bug, please check the [Issues page](https://github.com/Neil-urk12/totoo-ba-backend/issues) to see if it's already been reported. If not, open a new issue and use the "Bug Report" template.**

## Open a Pull Request (PR)

Open a Pull Request from your branch to the [main] branch of the original repository. [Pull Request](https://github.com/Neil-urk12/totoo-ba-backend/pulls)

Provide a clear title and description of your changes in the PR template. Reference the relevant issue (e.g., Closes #123).

Be ready to participate in a review and make requested changes!

## Commit Conventions

We follow the [Conventional Commits](https://www.conventionalcommits.org/) specification for commit messages. This helps maintain a clear history and enables automatic changelog generation.

### Format

```
<type>(<scope>): <subject>

<body>

<footer>
```

### Type
Must be one of the following:
- **feat**: A new feature
- **fix**: A bug fix
- **docs**: Documentation only changes
- **style**: Changes that don't affect code meaning (formatting, missing semicolons, etc.)
- **refactor**: Code change that neither fixes a bug nor adds a feature
- **perf**: Code change that improves performance
- **test**: Adding or updating tests
- **chore**: Changes to build process, dependencies, or tools

### Scope
Optional. The scope should specify what part of the codebase is affected (e.g., `api`, `database`, `auth`).

### Subject
- Use imperative, present tense: "add" not "added" or "adds"
- Don't capitalize first letter
- No period (.) at the end
- Limit to 50 characters

### Examples
```
feat(api): add user authentication endpoint
fix(database): resolve connection pool leak
docs: update installation instructions
refactor(services): simplify product verification logic
``` 
