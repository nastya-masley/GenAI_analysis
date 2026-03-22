
## Workflow Rules

1. Every code change after CLaude plan function was used must be done on a feature branch:
   - Branch name format: `feature/<short-description>`
   - Base branch used to create the feature branch must be recorded in this README.md file of feature branch.

2. For every significant change in `src/` or `app/`:
   - Update `CLAUDE.md` to reflect the current overall design / constraints.
   - Update `PLAN.md` to reflect the current implementation plan and completed steps.

3. For each new branch/plan:
   - Create a new git branch from the appropriate base branch.
   - Add an entry to this README describing:
     - Branch name
     - Base branch
     - Short description / scope of the feature implemented or to be implemented on  this branch.

4. Commit rules:
   - Every change must be committed to git history.
   - Commit message should describe:
     - What changed in the source code.
     - Whether `CLAUDE.md` and/or `PLAN.md` were updated.

