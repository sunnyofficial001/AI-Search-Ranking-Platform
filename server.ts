/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 *
 * Express Server — API Gateway Layer
 * ====================================
 * Serves the React SPA (via Vite middleware in dev, static files in prod)
 * and acts as an API gateway that proxies all /api/* requests to the
 * Python FastAPI backend running at PYTHON_API_URL (default: http://localhost:8000).
 *
 * Routes handled locally (NOT proxied):
 *   GET  /api/health         — Express-level liveness check
 *   POST /api/gemini/expand  — Uses @google/genai (Node.js SDK, key in process.env)
 *
 * All other /api/* routes are proxied to FastAPI with response-key mapping
 * where needed (snake_case → camelCase for React component compatibility).
 */

import express, { Request, Response, NextFunction } from 'express';
import path from 'path';
import { fileURLToPath } from 'url';
import { createServer as createViteServer } from 'vite';
import { GoogleGenAI } from '@google/genai';
import dotenv from 'dotenv';

dotenv.config();

// ESM / CJS compatibility — avoid TypeScript temporal-dead-zone error
// by capturing potential globals before const declarations.
const _globalFilename: string | undefined = (globalThis as any).__filename;
const _globalDirname:  string | undefined = (globalThis as any).__dirname;
const __filename = _globalFilename ?? fileURLToPath(import.meta.url);
const __dirname  = _globalDirname  ?? path.dirname(__filename);

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

const PORT            = parseInt(process.env.PORT ?? '3000', 10);
const PYTHON_API_URL  = (process.env.PYTHON_API_URL ?? 'http://localhost:8000').replace(/\/$/, '');

// ---------------------------------------------------------------------------
// Gemini client (optional — graceful no-op if key not provided)
// ---------------------------------------------------------------------------

let ai: GoogleGenAI | null = null;
try {
  if (process.env.GEMINI_API_KEY) {
    ai = new GoogleGenAI({
      apiKey: process.env.GEMINI_API_KEY,
      httpOptions: { headers: { 'User-Agent': 'aistudio-build' } },
    });
  }
} catch (err) {
  console.warn('[server] Gemini init failed:', err);
}

// ---------------------------------------------------------------------------
// Proxy helper
// ---------------------------------------------------------------------------

/**
 * Forward a request to the Python FastAPI backend and return the JSON response.
 * On network error it throws an Error with a descriptive message.
 */
async function proxyToFastAPI(
  method: string,
  path: string,
  body?: unknown,
  headers?: Record<string, string>,
): Promise<{ status: number; data: unknown }> {
  const url = `${PYTHON_API_URL}${path}`;
  const init: RequestInit = {
    method,
    headers: { 'Content-Type': 'application/json', ...(headers ?? {}) },
    ...(body !== undefined ? { body: JSON.stringify(body) } : {}),
  };

  let resp: globalThis.Response;
  try {
    resp = await fetch(url, init);
  } catch (err: any) {
    throw new Error(
      `Cannot reach Python backend at ${PYTHON_API_URL}. ` +
      `Start it with: python -m uvicorn backend.main:app --port 8000 (${err.message})`
    );
  }

  const contentType = resp.headers.get('content-type') ?? '';
  const data = contentType.includes('application/json') ? await resp.json() : await resp.text();
  return { status: resp.status, data };
}

// ---------------------------------------------------------------------------
// Response-shape adapters
// (FastAPI returns snake_case; React components expect camelCase in some fields)
// ---------------------------------------------------------------------------

function adaptSearchResult(item: any): any {
  const productId = item.doc_id ?? item.productId ?? '';
  const title = item.title ?? '';
  const category = item.category ?? 'Unknown';
  const originalRank = item.original_rank ?? item.originalRank ?? 0;
  const finalRank = item.final_rank ?? item.finalRank ?? 0;
  const relevanceLabel = item.relevance_label ?? item.relevanceLabel ?? 0;
  const scores = item.scores ?? {};
  const features = item.features ?? {};

  return {
    product: {
      id: productId,
      title,
      description: item.description ?? '',
      category,
      popularity: features.popularity ?? 0,
      ctr: features.ctr ?? 0,
      freshness: features.freshness ?? 0,
      engagement: features.engagement ?? 0,
      createdAt: item.created_at ?? new Date().toISOString(),
    },
    features: {
      bm25: features.bm25 ?? 0,
      tfidf: features.tfidf ?? 0,
      cosine: features.cosine ?? 0,
      queryLength: features.queryLength ?? 0,
      docLength: features.docLength ?? 0,
      popularity: features.popularity ?? 0,
      ctr: features.ctr ?? 0,
      freshness: features.freshness ?? 0,
      engagement: features.engagement ?? 0,
    },
    rankingScores: {
      pointwise: scores.pointwise ?? 0,
      pairwise: scores.pairwise ?? 0,
      listwise: scores.listwise ?? 0,
    },
    finalRank,
    originalRank,
    relevanceLabel,
    explanation: item.explanation ?? {},
  };
}

function adaptRecommendResult(item: any): any {
  const productId = item.item_id ?? item.productId ?? '';
  return {
    product: {
      id: productId,
      title: item.title ?? '',
      description: item.description ?? '',
      category: item.category ?? 'Unknown',
      popularity: item.popularity ?? 0,
      ctr: item.ctr ?? 0,
      freshness: item.freshness ?? 0,
      engagement: item.engagement ?? 0,
      createdAt: item.created_at ?? new Date().toISOString(),
    },
    score: item.score ?? 0,
    type: item.recommendation_type ?? item.type ?? 'hybrid',
    breakdown: item.explanation ?? item.breakdown ?? '',
  };
}

// ---------------------------------------------------------------------------
// Express application
// ---------------------------------------------------------------------------

const app = express();
app.use(express.json({ limit: '2mb' }));

// ---------------------------------------------------------------------------
// 1. Health — Express-level (does NOT require Python backend to be running)
// ---------------------------------------------------------------------------

app.get('/api/health', async (_req, res) => {
  // Also attempt to reach the FastAPI health endpoint
  try {
    const { status, data } = await proxyToFastAPI('GET', '/api/v1/health');
    res.status(status).json({
      express: 'healthy',
      fastapi: data,
    });
  } catch {
    res.json({
      express: 'healthy',
      fastapi: 'unreachable — start the Python backend on port 8000',
    });
  }
});

// ---------------------------------------------------------------------------
// 2. Gemini query expansion — handled in Node.js (uses @google/genai)
// ---------------------------------------------------------------------------

app.post('/api/gemini/expand', async (req, res) => {
  const { query, mode } = req.body ?? {};
  if (!query) {
    return res.status(400).json({ error: 'Missing query parameter' });
  }

  if (!ai) {
    return res.json({
      expanded: `${query} premium quality`,
      explanation: 'Gemini API key not configured. Using rule-based expansion fallback.',
    });
  }

  const systemPrompt = `You are a Principal Machine Learning Engineer at Amazon Search.
Analyze the query: "${query}" for mode: "${mode}".
Generate a JSON response with two keys:
1. "expanded": space-separated string of optimized BM25 query expansion tokens.
2. "explanation": concise 2-sentence engineering brief on why these synonyms improve recall.
Return ONLY valid JSON. No markdown code wraps.`;

  try {
    const response = await ai.models.generateContent({
      model: 'gemini-2.5-flash',
      contents: systemPrompt,
      config: { responseMimeType: 'application/json' },
    });
    const parsed = JSON.parse(response.text?.trim() ?? '{}');
    res.json(parsed);
  } catch (err: any) {
    res.json({
      expanded: `${query} premium quality`,
      explanation: `Gemini parse error: ${err.message}`,
    });
  }
});

// ---------------------------------------------------------------------------
// 3. Search — proxy to FastAPI /api/v1/v2/search
// ---------------------------------------------------------------------------

app.post('/api/search', async (req, res) => {
  try {
    const { query, weights, algorithm, page, page_size, user_id } = req.body ?? {};
    const { status, data } = await proxyToFastAPI('POST', '/api/v1/v2/search', {
      query:       query ?? '',
      algorithm:   algorithm ?? 'ensemble',
      page:        page ?? 1,
      page_size:   page_size ?? 10,
      user_id:     user_id ?? null,
      weights:     weights ?? null,
    });

    if (typeof data !== 'object' || data === null) {
      return res.status(status as number).json(data);
    }

    const raw = data as any;
    res.status(status as number).json({
      query: raw.query,
      expandedQuery: raw.expandedQuery ?? raw.query,
      algorithm: raw.algorithm,
      results: (raw.results ?? []).map(adaptSearchResult),
      pipeline_metadata: raw.pipeline_metadata,
      evaluation_metrics: raw.evaluation_metrics,
    });
  } catch (err: any) {
    res.status(503).json({ error: err.message });
  }
});

// ---------------------------------------------------------------------------
// 4. Recommendations — proxy to FastAPI /api/v1/v2/recommend
// ---------------------------------------------------------------------------

app.post('/api/recommend', async (req, res) => {
  try {
    const { userId, productId, type, hybridWeight, epochs, latentDim, lr } = req.body ?? {};
    const { status, data } = await proxyToFastAPI('POST', '/api/v1/recommend', {
      userId: userId ?? 'user-1',
      productId: productId ?? null,
      type: type ?? 'hybrid',
      hybridWeight: hybridWeight ?? 0.5,
      epochs: epochs ?? 30,
      latentDim: latentDim ?? 3,
      lr: lr ?? 0.05,
    });

    if (typeof data !== 'object' || data === null) {
      return res.status(status as number).json(data);
    }

    const raw = data as any;
    res.status(status as number).json({
      type: raw.type ?? type,
      results: (raw.results ?? []).map(adaptRecommendResult),
    });
  } catch (err: any) {
    res.status(503).json({ error: err.message });
  }
});

// ---------------------------------------------------------------------------
// 5. Explainability — proxy to FastAPI /api/v1/explain
// ---------------------------------------------------------------------------

app.post('/api/explain', async (req, res) => {
  try {
    const { productId, query } = req.body ?? {};
    const { status, data } = await proxyToFastAPI('POST', '/api/v1/explain', {
      productId:  productId,
      query:      query ?? '',
    });
    res.status(status as number).json(data);
  } catch (err: any) {
    res.status(503).json({ error: err.message });
  }
});

// ---------------------------------------------------------------------------
// 6. Experiments list — proxy to FastAPI /api/v1/experiments
// ---------------------------------------------------------------------------

app.get('/api/experiments', async (_req, res) => {
  try {
    const { status, data } = await proxyToFastAPI('GET', '/api/v1/experiments');
    res.status(status as number).json(data);
  } catch (err: any) {
    res.status(503).json({ error: err.message });
  }
});

// ---------------------------------------------------------------------------
// 7. Delete experiment run — proxy to FastAPI
// ---------------------------------------------------------------------------

app.delete('/api/experiments/:runId', async (req, res) => {
  try {
    const { runId } = req.params;
    const { status, data } = await proxyToFastAPI('DELETE', `/api/v1/experiments/${runId}`);
    res.status(status as number).json(data);
  } catch (err: any) {
    res.status(503).json({ error: err.message });
  }
});

// ---------------------------------------------------------------------------
// 8. Feature store catalog — proxy to FastAPI /api/v1/feature-store/catalog
// ---------------------------------------------------------------------------

app.get('/api/feature-store', async (_req, res) => {
  try {
    const { status, data } = await proxyToFastAPI('GET', '/api/v1/feature-store/catalog');
    if (typeof data !== 'object' || data === null) {
      return res.status(status as number).json(data);
    }

    // Adapt feature catalog into the shape expected by DatasetFeatures.tsx
    const raw = data as any;
    const catalog: any[] = raw.catalog ?? [];

    // Group features by source to build a per-product-like view
    // The frontend expects: [{ productId, category, features: { bm25, tfidf, ... } }]
    // Since catalog is feature *specs*, we return them directly and let DatasetFeatures
    // render the spec view (the component checks for productId, so we adapt minimally)
    const adapted = catalog.map((spec: any, i: number) => ({
      productId:  spec.name,
      category:   spec.source ?? 'feature_store',
      features: {
        bm25:       parseFloat((i * 0.35 + 0.1).toFixed(2)),  // illustrative slot value
        tfidf:      parseFloat((i * 0.18 + 0.05).toFixed(2)),
        cosine:     parseFloat((0.9 - i * 0.04).toFixed(2)),
        ctr:        parseFloat((0.05 + i * 0.01).toFixed(3)),
        engagement: parseFloat((4.5 - i * 0.03).toFixed(1)),
      },
      description: spec.description,
      tags:        spec.tags,
      version:     spec.version,
    }));

    res.status(status as number).json(adapted);
  } catch (err: any) {
    res.status(503).json({ error: err.message });
  }
});

// ---------------------------------------------------------------------------
// 9. Rank training — proxy to FastAPI /api/v1/rank/train
// ---------------------------------------------------------------------------

app.post('/api/rank/train', async (req, res) => {
  try {
    const { status, data } = await proxyToFastAPI('POST', '/api/v1/rank/train', req.body ?? {});
    res.status(status as number).json(data);
  } catch (err: any) {
    res.status(503).json({ error: err.message });
  }
});

// ---------------------------------------------------------------------------
// 10. Data Pipeline run — proxy to FastAPI /api/v1/pipeline/run
// ---------------------------------------------------------------------------

app.post('/api/pipeline/run', async (req, res) => {
  try {
    const { status, data } = await proxyToFastAPI('POST', '/api/v1/pipeline/run', req.body ?? {});
    res.status(status as number).json(data);
  } catch (err: any) {
    res.status(503).json({ error: err.message });
  }
});

// ---------------------------------------------------------------------------
// Vite / Static file serving
// ---------------------------------------------------------------------------

async function startServer() {
  if (process.env.NODE_ENV !== 'production') {
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: 'spa',
    });
    app.use(vite.middlewares);
  } else {
    const distPath = path.join(process.cwd(), 'dist');
    app.use(express.static(distPath));
    app.get('*', (_req, res) => {
      res.sendFile(path.join(distPath, 'index.html'));
    });
  }

  app.listen(PORT, '0.0.0.0', () => {
    console.log(`\n🚀  AI Search Platform (Express gateway) → http://localhost:${PORT}`);
    console.log(`🐍  Proxying /api/* → FastAPI backend at  ${PYTHON_API_URL}`);
    console.log(`📖  FastAPI docs (when running):          ${PYTHON_API_URL}/docs\n`);
  });
}

startServer();
