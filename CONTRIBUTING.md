# Contributing Guidelines

Thank you for your interest in contributing to the **AI Search & Ranking Platform**!

---

## 🛠️ Development Setup

1. **Fork & Clone** the repository:
   ```bash
   git clone https://github.com/sunnyofficial001/AI-Search-Ranking-Platform.git
   cd AI-Search-Ranking-Platform
   ```

2. **Setup Python Virtual Environment**:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # or .\.venv\Scripts\activate on Windows
   pip install -r backend/requirements.txt
   ```

3. **Install Frontend Dependencies**:
   ```bash
   npm install
   ```

4. **Run Test Suite**:
   ```bash
   python -m pytest tests/unit/ -v
   npm run lint
   ```

---

## 📐 Coding & Quality Standards

- **Python**: Follow PEP 8 guidelines. Keep docstrings clear and preserve existing API parameters.
- **TypeScript / React**: Use functional components with strict TypeScript types (`npm run lint`).
- **Tests**: Write unit tests for all new ranking algorithms, data transformations, or API routes under `tests/`.
- **Commit Messages**: Follow standard conventional commit format:
  - `feat: add neural reranker module`
  - `fix: resolve Redis connection timeout handling`
  - `docs: update system design architecture`

---

## 🔒 Security & Secrets

Do NOT commit `.env` files, API keys, passwords, or binary databases. Always verify `git status` before pushing changes.
