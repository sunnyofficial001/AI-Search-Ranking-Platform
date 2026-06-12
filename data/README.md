# MSLR-WEB10K Dataset

This directory should contain the **Microsoft Learning to Rank (MSLR-WEB10K)** dataset.

## Download Instructions

1. Visit the official dataset page:  
   **https://www.microsoft.com/en-us/research/project/mslr/**

2. Download `MSLR-WEB10K.zip`

3. Extract and place the Fold1 files here:
   ```
   data/
   └── Fold1/
       ├── train.txt   (~400MB)
       ├── vali.txt    (~140MB)
       └── test.txt    (~140MB)
   ```

## Format

The files are in **SVMLight / LETOR** format:
```
<relevance> qid:<query_id> 1:<feat_1> 2:<feat_2> ... 136:<feat_136> # <comment>
```

- **Relevance**: 0 (irrelevant) → 4 (perfectly relevant)
- **Query ID**: groups documents belonging to the same query
- **Features 1–136**: Various IR, text similarity, and link analysis features

## Dataset Statistics (Fold1)

| Split | Documents | Queries |
|-------|-----------|---------|
| train.txt | 723,412 (full) / 150,000 (sampled) | 6,000 |
| vali.txt | 235,259 | 2,000 |
| test.txt | 241,521 | 2,000 |

The large files are excluded from this repository via `.gitignore` due to size (~800MB total).
