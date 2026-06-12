# SECURITY

This project follows secure handling of secrets and credentials. Do NOT commit any real API keys, passwords, tokens, or private keys to source control. Use environment variables and CI/CD secret stores.

Guidelines

- Local secrets: create a local `.env` file (not committed) and add it to `.gitignore`. Copy from `.env.example` and replace placeholders.
- CI/CD: store secrets in GitHub Actions Secrets, Azure Key Vault, Railway/Heroku config vars, or other secure secret manager. Reference them in workflows as `${{ secrets.NAME }}`.
- Docker: avoid hardcoding credentials in `docker-compose.yml`. Use environment variable substitution or Docker secrets.
- Rotating keys: rotate any leaked keys immediately and update downstream services.
- Secret scanning: this repository includes a detect-secrets baseline `.secrets.baseline` for local pre-commit scanning. Run `detect-secrets scan` locally to produce/update the baseline.

Quick commands

```bash
# Install detect-secrets: pip install detect-secrets
detect-secrets scan > .secrets.baseline
detect-secrets audit .secrets.baseline
detect-secrets-hook --baseline .secrets.baseline install

# Or use git-secrets
git secrets --install
git secrets --register-aws
git secrets --scan
```

If you discover a secret in the repository history, remove it and rotate the secret. Use `git filter-repo` or contact GitHub support for help removing secrets from history.
