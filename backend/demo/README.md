Demo generator and runner

This folder provides a tiny synthetic dataset generator and a demo runner that posts
a sample search request to the running dev server (Express/Vite server at port 3000).

Usage

1. Generate demo data only:

```bash
python backend/demo/generate_demo_data.py
```

2. Generate data and run a sample search against a running dev server:

```bash
python backend/demo/run_demo.py
```

Notes
- The generated files will be written under `data/Fold1/`.
- The demo runner uses only Python standard library (no extra dependencies).
