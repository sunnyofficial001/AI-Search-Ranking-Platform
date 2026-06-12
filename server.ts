/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import express from 'express';
import path from 'path';
import { fileURLToPath } from 'url';
import { createServer as createViteServer } from 'vite';
import { GoogleGenAI } from '@google/genai';
import dotenv from 'dotenv';

// Import our mathematical and simulation engines
import {
  performIntegratedSearch,
  trainRankNet,
  trainLambdaMART,
  getRecommendContentBased,
  getCollaborativeFiltering,
  trainMatrixFactorization,
  getHybridRecommendations,
  calculateSHAPForProduct,
  MOCK_PRODUCTS,
  USER_RATING_MATRIX,
  INSTALLED_RUNS
} from './src/engines.js';

dotenv.config();

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Initialize Gemini client using server-only process.env key with mandatory custom User-Agent
let ai: GoogleGenAI | null = null;
try {
  if (process.env.GEMINI_API_KEY) {
    ai = new GoogleGenAI({
      apiKey: process.env.GEMINI_API_KEY,
      httpOptions: {
        headers: {
          'User-Agent': 'aistudio-build',
        }
      }
    });
  }
} catch (err) {
  console.warn("Failed to initialize Gemini. Check process.env.GEMINI_API_KEY.", err);
}

const app = express();
const PORT = 3000;

app.use(express.json());

// In-Memory state representing modified weights / dynamic experiment registries
let activeExperimentRuns = [...INSTALLED_RUNS];
let matrixFactorizationCache: any = null;

// ==========================================
// API ENDPOINTS
// ==========================================

// 1. Health Status Check
app.get('/api/health', (req, res) => {
  res.json({
    status: 'healthy',
    timestamp: new Date().toISOString(),
    database: 'Seeded memory store',
    dataset: 'MSLR-WEB10K preprocessed simulation'
  });
});

// 2. Intelligent Search retrieval (BM25 -> normalizing -> candidate expansion)
app.post('/api/search', (req, res) => {
  const { query, weights } = req.body;
  if (!query) {
    return res.status(400).json({ error: 'Missing query parameter in request body' });
  }

  const response = performIntegratedSearch(query, weights);
  res.json(response);
});

// 3. Gemini query optimization & developer context advisor
app.post('/api/gemini/expand', async (req, res) => {
  const { query, mode } = req.body;
  if (!ai) {
    return res.json({
      expanded: `${query} plus premium match (Gemini offline fallback)`,
      explanation: "Gemini server-side API Key is not configured yet. Configure GEMINI_API_KEY in Settings."
    });
  }

  const systemPrompt = `You are a Principal Machine Learning Engineer at Amazon Search. 
Analyze the query: "${query}" for mode: "${mode}". 
Generate a JSON response conforming to the schema of having two keys: 
1. "expanded": a space-separated string of optimized search token expansions representing BM25 query expansion, TF-IDF synonyms, or category tags.
2. "explanation": a concise engineering brief (2 sentences) detailing why you added these synonyms to improve recall under MSLR-WEB10K distribution.
Return ONLY valid JSON. No markdown code wraps.`;

  try {
    const response = await ai.models.generateContent({
      model: "gemini-3.5-flash",
      contents: systemPrompt,
      config: {
        responseMimeType: "application/json"
      }
    });
    
    const parsed = JSON.parse(response.text?.trim() || "{}");
    res.json(parsed);
  } catch (err: any) {
    res.json({
      expanded: `${query} custom recommendation`,
      explanation: `Failed query parsing: ${err.message}`
    });
  }
});

// 4. Learning to Rank Train Loops comparison (Pointwise, Pairwise RankNet, Listwise LambdaMART)
app.post('/api/rank/train', (req, res) => {
  const { algorithm, learningRate, epochs, nEstimators } = req.body;

  let history: any[] = [];
  const runId = `run-${Math.floor(1000 + Math.random() * 9000)}`;

  if (algorithm === 'pairwise_ranknet') {
    history = trainRankNet(learningRate || 0.02, epochs || 50);
    const last = history[history.length - 1];
    const newRun = {
      runId,
      name: `train_ranknet_${runId}`,
      timestamp: new Date().toISOString(),
      algorithm: 'pairwise_ranknet' as const,
      parameters: { learning_rate: learningRate || 0.02, epochs: epochs || 50 },
      metrics: {
        ndcg5: 0.76,
        ndcg10: 0.82,
        map: 0.73,
        mrr: 0.79,
        precision5: 0.66,
        recall5: 0.76,
        finalLoss: last?.loss || 0.32
      },
      status: 'SUCCESS' as const
    };
    activeExperimentRuns.unshift(newRun);
    res.json({ run: newRun, history });

  } else if (algorithm === 'listwise_lambdamart') {
    history = trainLambdaMART(nEstimators || 20, learningRate || 0.1);
    const last = history[history.length - 1];
    const newRun = {
      runId,
      name: `train_lambdamart_${runId}`,
      timestamp: new Date().toISOString(),
      algorithm: 'listwise_lambdamart' as const,
      parameters: { n_estimators: nEstimators || 20, learning_rate: learningRate || 0.1 },
      metrics: {
        ndcg5: last?.ndcg5 || 0.88,
        ndcg10: last?.ndcg10 || 0.93,
        map: 0.84,
        mrr: 0.90,
        precision5: 0.78,
        recall5: 0.82,
        finalLoss: last?.loss || 0.14
      },
      status: 'SUCCESS' as const
    };
    activeExperimentRuns.unshift(newRun);
    res.json({ run: newRun, history });

  } else {
    // Pointwise baseline run registration
    const newRun = {
      runId,
      name: `pointwise_baseline_${runId}`,
      timestamp: new Date().toISOString(),
      algorithm: 'pointwise' as const,
      parameters: { static_weights: 'MSLR-default' },
      metrics: {
        ndcg5: 0.69,
        ndcg10: 0.75,
        map: 0.66,
        mrr: 0.71,
        precision5: 0.61,
        recall5: 0.73
      },
      status: 'SUCCESS' as const
    };
    activeExperimentRuns.unshift(newRun);
    res.json({ run: newRun, history: [{ step: 1, loss: 0.52 }] });
  }
});

// 5. Recommendations API with Dynamic parameters
app.post('/api/recommend', (req, res) => {
  const { userId, productId, type, hybridWeight, epochs, latentDim, lr } = req.body;
  const activeUser = userId || 'user-1';

  if (type === 'content') {
    const list = getRecommendContentBased(productId || 'prod-1');
    return res.json({ type, results: list });
  }

  if (type === 'collaborative') {
    const list = getCollaborativeFiltering(activeUser);
    return res.json({ type, results: list });
  }

  if (type === 'matrix_factorization') {
    // Check if training requested or retrieve existing matrix factorization projection
    const ep = epochs || 30;
    const dim = latentDim || 3;
    const lrate = lr || 0.05;

    const trained = trainMatrixFactorization(ep, dim, lrate);
    matrixFactorizationCache = trained;

    // Use predicted matrix to generate recommendations for active user
    const userLatent = trained.pMatrix[activeUser];
    const results: any[] = [];

    MOCK_PRODUCTS.forEach(p => {
      // Avoid raw items already rated by user to focus on novel discovery
      const rated = USER_RATING_MATRIX[activeUser][p.id];
      if (rated && rated > 0) return;

      const prodLatent = trained.qMatrix[p.id];
      if (userLatent && prodLatent) {
        // Dot product projection
        let score = 0;
        for (let k = 0; k < dim; k++) {
          score += userLatent[k] * prodLatent[k];
        }
        results.push({
          product: p,
          score: parseFloat((score / 5.0).toFixed(3)),
          type: 'matrix_factorization',
          breakdown: `Factored prediction dot product: ${score.toFixed(2)}.`
        });
      }
    });

    results.sort((a, b) => b.score - a.score);

    return res.json({
      type,
      losses: trained.losses,
      results: results.slice(0, 4),
      pMatrix: trained.pMatrix,
      qMatrix: trained.qMatrix
    });
  }

  // Hybrid match default
  const list = getHybridRecommendations(activeUser, productId || 'prod-1', 4, hybridWeight ?? 0.5);
  res.json({ type: 'hybrid', results: list });
});

// 6. Explainable AI: SHAP Coalitions endpoint
app.post('/api/explain', (req, res) => {
  const { productId, query } = req.body;
  const prod = MOCK_PRODUCTS.find(p => p.id === productId);
  if (!prod) {
    return res.status(404).json({ error: "Product not registered in database." });
  }

  const shapVal = calculateSHAPForProduct(prod, query || "smart speaker ANC controller");
  res.json(shapVal);
});

// 7. MLflow registry list
app.get('/api/experiments', (req, res) => {
  res.json({
    runs: activeExperimentRuns,
    modelRegistry: [
      { modelName: "LambdaMART-LTR-Production", version: "v1.2.0", accuracy: "0.88 NDCG@5", status: "Active" },
      { modelName: "MatrixFactorization-Recs", version: "v2.1-latent", accuracy: "0.85 Precision@5", status: "Active" },
      { modelName: "RankNet-Pairwise", version: "v0.9.1", accuracy: "0.75 NDCG@5", status: "Staging" }
    ]
  });
});

// 8. Delete experiment run
app.delete('/api/experiments/:runId', (req, res) => {
  const { runId } = req.params;
  activeExperimentRuns = activeExperimentRuns.filter(r => r.runId !== runId);
  res.json({ success: true, runId });
});

// 9. Initial Feature Store list
app.get('/api/feature-store', (req, res) => {
  const store = MOCK_PRODUCTS.map(p => {
    return {
      productId: p.id,
      title: p.title,
      category: p.category,
      features: {
        bm25: parseFloat((Math.random() * 3.5).toFixed(2)),
        tfidf: parseFloat((Math.random() * 2.1).toFixed(2)),
        cosine: parseFloat((Math.random() * 0.95).toFixed(2)),
        popularity: p.popularity,
        ctr: p.ctr,
        freshness: p.freshness,
        engagement: p.engagement
      }
    };
  });
  res.json(store);
});

// ==========================================
// VITE DEV SERVER / PRODUCTION ENTRY POINT
// ==========================================

async function startServer() {
  if (process.env.NODE_ENV !== "production") {
    // Development Mode
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: "spa",
    });
    app.use(vite.middlewares);
  } else {
    // Production Mode
    const distPath = path.join(process.cwd(), 'dist');
    app.use(express.static(distPath));
    app.get('*', (req, res) => {
      res.sendFile(path.join(distPath, 'index.html'));
    });
  }

  app.listen(PORT, "0.0.0.0", () => {
    console.log(`AI Search Platform server running at http://localhost:${PORT}`);
  });
}

startServer();
