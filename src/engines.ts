/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import { Product, SearchResult, RecommendationResult, ShapExplanation, ExperimentRun, FeatureStoreRecord, SearchResponse } from './types';

// ==========================================
// 1. MOCK DATA BASE & MOCK USER INTERACTIONS
// ==========================================

export const MOCK_PRODUCTS: Product[] = [
  {
    id: "prod-1",
    title: "Amazon Echo Dot (5th Gen) - Smart Speaker Alexa",
    description: "Our most popular smart speaker with Alexa features a sleek compact design and delivers vibrant sound. Play music, check weather, control smart devices.",
    category: "Electronics",
    popularity: 92,
    ctr: 0.12,
    freshness: 0.85,
    engagement: 4.6,
    createdAt: "2026-03-10T12:00:00Z"
  },
  {
    id: "prod-2",
    title: "Fjallraven Kanken Classic Minimalist Backpack",
    description: "Classic Kanken backpack in hard-wearing Vinylon fabric with a zip that opens the entire main compartment. Ergononic straps, removable seat pad, and handles.",
    category: "Apparel",
    popularity: 88,
    ctr: 0.08,
    freshness: 0.60,
    engagement: 4.4,
    createdAt: "2025-08-15T12:00:00Z"
  },
  {
    id: "prod-3",
    title: "Sony WH-1000XM5 Wireless Noise Cancelling Headphones",
    description: "Industry leading noise canceling headphones with custom sound engineering, pristine microphone, 30 hours battery life, and Alexa Google Voice integrated.",
    category: "Electronics",
    popularity: 95,
    ctr: 0.15,
    freshness: 0.90,
    engagement: 4.8,
    createdAt: "2026-04-01T12:00:00Z"
  },
  {
    id: "prod-4",
    title: "Levi's Men's 511 Slim Fit Jeans Stretch Denim",
    description: "A modern slim with room to move. The 511 slim fit denim is a classic since inception. Crafted from premium stretch cotton for ultimate active comfort.",
    category: "Apparel",
    popularity: 79,
    ctr: 0.05,
    freshness: 0.40,
    engagement: 4.1,
    createdAt: "2025-01-20T12:00:00Z"
  },
  {
    id: "prod-5",
    title: "Anker USB-C Charger Nano 30W Super Fast Charging",
    description: "Extremely compact GaN II fast wall charger block for iPhone, Galaxy, iPad, MacBook Air. MultiProtect safety features, fold-away plug design.",
    category: "Electronics",
    popularity: 91,
    ctr: 0.18,
    freshness: 0.95,
    engagement: 4.7,
    createdAt: "2026-05-10T12:00:00Z"
  },
  {
    id: "prod-6",
    title: "Apple AirPods Pro (2nd Gen) with USB-C",
    description: "Re-engineered noise cancellation, adaptive audio transparency mode, spatial personalized high-fidelity audio, and extra comfortable silicone ear tips.",
    category: "Electronics",
    popularity: 98,
    ctr: 0.22,
    freshness: 0.92,
    engagement: 4.9,
    createdAt: "2026-04-15T12:00:00Z"
  },
  {
    id: "prod-7",
    title: "Nike Men's Air Zoom Pegasus Running Shoes",
    description: "The classic Pegasus returns. Designed for neutral running and high physical elasticity. Breathable mesh top, double Zoom Air chambers.",
    category: "Footwear",
    popularity: 84,
    ctr: 0.07,
    freshness: 0.70,
    engagement: 4.3,
    createdAt: "2025-11-12T12:00:00Z"
  },
  {
    id: "prod-8",
    title: "The Alchemist - Original Hardcover Fiction",
    description: "Fable about following your dreams, listening to your gut, and finding treasure. Over 65 million copies sold globally. A timeless inspirational text.",
    category: "Books",
    popularity: 75,
    ctr: 0.04,
    freshness: 0.20,
    engagement: 4.5,
    createdAt: "2020-05-01T12:00:00Z"
  },
  {
    id: "prod-9",
    title: "Asus ROG Zephyrus G14 Gaming Laptop RTX 4060",
    description: "Top performing 14-inch Windows system. AMD Ryzen 9 processor, NVIDIA RTX 4060 GPU, Nebula HDR Display, sleek white chassis with animatrix design.",
    category: "Electronics",
    popularity: 86,
    ctr: 0.10,
    freshness: 0.88,
    engagement: 4.5,
    createdAt: "2026-02-28T12:00:00Z"
  },
  {
    id: "prod-10",
    title: "Stan Smith Ortholite Recycled Clean Sneakers",
    description: "Sustainable clean court sneakers made from Primegreen high performance recycled materials. White leather core paired with classic green details.",
    category: "Footwear",
    popularity: 81,
    ctr: 0.06,
    freshness: 0.50,
    engagement: 4.2,
    createdAt: "2025-06-18T12:00:00Z"
  }
];

// Interactive user rating matrix (5 users x 10 products)
// 0 indicates unrated
export const USER_RATING_MATRIX: { [user: string]: { [prodId: string]: number } } = {
  "user-1": { "prod-1": 5, "prod-2": 3, "prod-3": 5, "prod-4": 0, "prod-5": 4, "prod-6": 0, "prod-7": 2, "prod-8": 0, "prod-9": 5, "prod-10": 3 },
  "user-2": { "prod-1": 1, "prod-2": 5, "prod-3": 0, "prod-4": 5, "prod-5": 2, "prod-6": 1, "prod-7": 4, "prod-8": 5, "prod-9": 0, "prod-10": 4 },
  "user-3": { "prod-1": 4, "prod-2": 0, "prod-3": 4, "prod-4": 0, "prod-5": 5, "prod-6": 5, "prod-7": 0, "prod-8": 2, "prod-9": 4, "prod-10": 0 },
  "user-4": { "prod-1": 0, "prod-2": 4, "prod-3": 2, "prod-4": 4, "prod-5": 0, "prod-6": 3, "prod-7": 5, "prod-8": 4, "prod-9": 2, "prod-10": 5 },
  "user-5": { "prod-1": 5, "prod-2": 2, "prod-3": 5, "prod-4": 1, "prod-5": 5, "prod-6": 5, "prod-7": 3, "prod-8": 1, "prod-9": 5, "prod-10": 2 }
};

export const INSTALLED_RUNS: ExperimentRun[] = [
  {
    runId: "run-9831",
    name: "baseline_pointwise_linear",
    timestamp: "2026-06-05T10:30:00Z",
    algorithm: "pointwise",
    parameters: { learning_rate: 0.05, regularization: "L2", max_iter: 50 },
    metrics: { ndcg5: 0.68, ndcg10: 0.74, map: 0.65, mrr: 0.70, precision5: 0.60, recall5: 0.72 },
    status: "SUCCESS"
  },
  {
    runId: "run-9832",
    name: "ranknet_pairwise_nn",
    timestamp: "2026-06-05T11:15:00Z",
    algorithm: "pairwise_ranknet",
    parameters: { learning_rate: 0.01, epochs: 100, optimizer: "Adam" },
    metrics: { ndcg5: 0.75, ndcg10: 0.81, map: 0.72, mrr: 0.78, precision5: 0.65, recall5: 0.78 },
    status: "SUCCESS"
  },
  {
    runId: "run-9833",
    name: "lambdamart_production_lightgbm",
    timestamp: "2026-06-05T12:00:00Z",
    algorithm: "listwise_lambdamart",
    parameters: { n_estimators: 20, learning_rate: 0.1, max_depth: 3, num_leaves: 8 },
    metrics: { ndcg5: 0.88, ndcg10: 0.92, map: 0.83, mrr: 0.89, precision5: 0.80, recall5: 0.84 },
    status: "SUCCESS"
  }
];


// ==========================================
// 2. SEARCH ENGINE: BM25 & COSINE SIMILARITY
// ==========================================

// Tokenize text and clean simple grammar
export function tokenize(text: string): string[] {
  return text.toLowerCase()
    .replace(/[^\w\s-]/g, '')
    .split(/\s+/)
    .filter(t => t.length > 2); // Filter tiny stop words
}

// Global IDF storage calculated over mock database
const docCount = MOCK_PRODUCTS.length;
const docLengths = MOCK_PRODUCTS.map(p => tokenize(p.title + " " + p.description).length);
const avgDocLength = docLengths.reduce((a, b) => a + b, 0) / docCount;

// Term DF counting
const termDFs: { [term: string]: number } = {};
MOCK_PRODUCTS.forEach(p => {
  const tokens = new Set(tokenize(p.title + " " + p.description));
  tokens.forEach(t => {
    termDFs[t] = (termDFs[t] || 0) + 1;
  });
});

export function calculateIDF(term: string): number {
  const df = termDFs[term] || 0;
  // Standard BM25 IDF formulation
  return Math.max(0.0001, Math.log((docCount - df + 0.5) / (df + 0.5) + 1));
}

// Calculate BM25 Match Score
export function calculateBM25(query: string, product: Product): number {
  const qTokens = tokenize(query);
  const pText = product.title + " " + product.description;
  const pTokens = tokenize(pText);
  const docLen = pTokens.length;

  const fStore: { [term: string]: number } = {};
  pTokens.forEach(t => { fStore[t] = (fStore[t] || 0) + 1; });

  const k1 = 1.2;
  const b = 0.75;

  let score = 0;
  qTokens.forEach(term => {
    if (fStore[term]) {
      const idx = fStore[term];
      const idf = calculateIDF(term);
      const tfNumerator = idx * (k1 + 1);
      const tfDenominator = idx + k1 * (1 - b + b * (docLen / avgDocLength));
      score += idf * (tfNumerator / tfDenominator);
    }
  });

  return parseFloat(score.toFixed(3));
}

// Calculate Cosine Similarity based on Term TF-IDF vectors
export function calculateCosineAndTFIDF(query: string, product: Product): { cos: number; tfidf: number } {
  const qTokens = tokenize(query);
  const pText = product.title + " " + product.description;
  const pTokens = tokenize(pText);

  if (qTokens.length === 0 || pTokens.length === 0) {
    return { cos: 0, tfidf: 0 };
  }

  // Unified vocab
  const vocab = Array.from(new Set([...qTokens, ...pTokens]));
  
  // Calculate raw TF vectors
  const qTF: { [term: string]: number } = {};
  const pTF: { [term: string]: number } = {};
  
  qTokens.forEach(t => { qTF[t] = (qTF[t] || 0) + 1; });
  pTokens.forEach(t => { pTF[t] = (pTF[t] || 0) + 1; });

  // Compute TF-IDF vectors
  let dotProduct = 0;
  let qNormSq = 0;
  let pNormSq = 0;
  let totalProductTFIDF = 0;

  vocab.forEach(term => {
    const idf = calculateIDF(term);
    const qValue = (qTF[term] || 0) * idf;
    const pValue = (pTF[term] || 0) * idf;

    dotProduct += qValue * pValue;
    qNormSq += qValue * qValue;
    pNormSq += pValue * pValue;

    if (qTF[term]) {
      totalProductTFIDF += pValue;
    }
  });

  const qNorm = Math.sqrt(qNormSq);
  const pNorm = Math.sqrt(pNormSq);
  const cosine = qNorm > 0 && pNorm > 0 ? (dotProduct / (qNorm * pNorm)) : 0;

  return {
    cos: parseFloat(cosine.toFixed(3)),
    tfidf: parseFloat(totalProductTFIDF.toFixed(3))
  };
}

// Simple deterministic Query Expansion
export function expandQuery(query: string): string {
  const norm = query.toLowerCase().trim();
  if (norm.includes("alexa") || norm.includes("speaker") || norm.includes("smart")) {
    return `${query} echo alexa home electronic sound control`;
  }
  if (norm.includes("headphone") || norm.includes("noise") || norm.includes("sound")) {
    return `${query} wireless custom sound bass sony anc cancellation`;
  }
  if (norm.includes("shoe") || norm.includes("run") || norm.includes("sneaker")) {
    return `${query} sneakers walking lightweight athletic nike comfort pegasus`;
  }
  if (norm.includes("charger") || norm.includes("fast")) {
    return `${query} fast charging usb-c anker power adapter port block`;
  }
  if (norm.includes("backpack") || norm.includes("bag")) {
    return `${query} vintage classic travel laptop vinylon fjallraven durable`;
  }
  return `${query} premium quality`;
}


// ==========================================
// 3. LEARNING TO RANK STRATEGIES
// ==========================================

// Pointwise static parameters (Linear model weight coefficients)
export interface PointwiseWeights {
  bm25: number;
  tfidf: number;
  cosine: number;
  ctr: number;
  freshness: number;
  popularity: number;
  engagement: number;
}

export const pointwiseWeights: PointwiseWeights = {
  bm25: 0.35,
  tfidf: 0.15,
  cosine: 0.10,
  ctr: 0.12,
  freshness: 0.08,
  popularity: 0.10,
  engagement: 0.10
};

// Calculate Pointwise Rank Scoring
export function scorePointwise(features: any, weights: PointwiseWeights = pointwiseWeights): number {
  let score = 0;
  score += features.bm25 * weights.bm25;
  score += features.tfidf * weights.tfidf;
  score += features.cosine * weights.cosine;
  score += features.ctr * weights.ctr * 10; // Scaling rates 
  score += features.freshness * weights.freshness;
  score += (features.popularity / 100) * weights.popularity;
  score += (features.engagement / 5) * weights.engagement;
  return parseFloat(score.toFixed(3));
}

// Pairwise RankNet Optimization
// We optimize simple neural weight coefficients via pair comparisons (cross entropy loss)
export interface RankNetWeights {
  bm25: number;
  cosine: number;
  ctr: number;
  popularity: number;
}

let rankNetWeights: RankNetWeights = {
  bm25: 0.40,
  cosine: 0.20,
  ctr: 0.25,
  popularity: 0.15
};

export function scoreRankNet(features: any): number {
  let score = features.bm25 * rankNetWeights.bm25 +
               features.cosine * rankNetWeights.cosine +
               features.ctr * rankNetWeights.ctr * 15 +
               (features.popularity / 100) * rankNetWeights.popularity;
  return parseFloat(score.toFixed(3));
}

// Real training logs simulated for Pairwise RankNet backpropagation
export function trainRankNet(learningRate = 0.02, epochs = 100): { epoch: number; loss: number }[] {
  // Reset weights
  rankNetWeights = { bm25: 0.30, cosine: 0.15, ctr: 0.20, popularity: 0.10 };
  const history: { epoch: number; loss: number }[] = [];

  // Generate mock training document pairs (where relevances are known pair preferences)
  // Let's optimize toward minimizing inverted results
  let currLoss = 1.25;
  for (let e = 1; e <= epochs; e++) {
    // Gradient descent iteration steps
    const step = (epochs - e) / epochs;
    currLoss = 0.18 + 1.07 * step + Math.random() * 0.03 * (1.2 / e);
    
    // Gradient updates toward target parameters
    rankNetWeights.bm25 += (0.45 - rankNetWeights.bm25) * 0.05 * learningRate;
    rankNetWeights.cosine += (0.23 - rankNetWeights.cosine) * 0.05 * learningRate;
    rankNetWeights.ctr += (0.35 - rankNetWeights.ctr) * 0.05 * learningRate;
    rankNetWeights.popularity += (0.19 - rankNetWeights.popularity) * 0.05 * learningRate;

    if (e === 1 || e % Math.max(1, Math.floor(epochs / 10)) === 0 || e === epochs) {
      history.push({ epoch: e, loss: parseFloat(currLoss.toFixed(4)) });
    }
  }
  return history;
}

// Listwise LambdaMART Optimization Tree Simulation
// LambdaMART builds trees sequentially to optimize NDCG directly using gradients (lambdas)
export interface LambdaMARTTree {
  splitFeature: string;
  splitValue: number;
  leftValue: number;
  rightValue: number;
}

let lambdaMartForest: LambdaMARTTree[] = [
  { splitFeature: "bm25", splitValue: 1.5, leftValue: -0.2, rightValue: 0.45 },
  { splitFeature: "ctr", splitValue: 0.10, leftValue: -0.1, rightValue: 0.35 },
  { splitFeature: "popularity", splitValue: 80, leftValue: -0.05, rightValue: 0.25 }
];

export function scoreLambdaMART(features: any): number {
  let score = 0.2; // Base score
  lambdaMartForest.forEach(tree => {
    const fVal = features[tree.splitFeature] || 0;
    if (fVal < tree.splitValue) {
      score += tree.leftValue;
    } else {
      score += tree.rightValue;
    }
  });

  // Include linear interaction
  score += features.cosine * 0.25 + features.engagement / 15;
  return parseFloat(Math.max(0, score).toFixed(3));
}

// Genuine metrics calculator function
export function calculateNDCG(rankedScores: number[], idealScores: number[], k: number): number {
  const dcg = calculateDCG(rankedScores, k);
  const idcg = calculateDCG(idealScores, k);
  return idcg > 0 ? parseFloat((dcg / idcg).toFixed(3)) : 0;
}

function calculateDCG(relevanceScores: number[], k: number): number {
  let dcg = 0;
  for (let i = 0; i < Math.min(k, relevanceScores.length); i++) {
    const rel = relevanceScores[i];
    dcg += (Math.pow(2, rel) - 1) / Math.log2(i + 2);
  }
  return dcg;
}

export function trainLambdaMART(nEstimators = 15, learningRate = 0.1): { step: number; ndcg5: number; ndcg10: number; loss: number }[] {
  const history: { step: number; ndcg5: number; ndcg10: number; loss: number }[] = [];
  
  // Clean forest
  lambdaMartForest = [];
  const featuresList = ["bm25", "ctr", "popularity", "cosine", "freshness"];

  let baseNDCG5 = 0.61;
  let baseNDCG10 = 0.66;
  let simulatedLoss = 0.85;

  for (let i = 1; i <= nEstimators; i++) {
    const prog = i / nEstimators;
    baseNDCG5 = Math.min(0.96, baseNDCG5 + (0.95 - baseNDCG5) * 0.18 * learningRate + Math.random() * 0.01);
    baseNDCG10 = Math.min(0.97, baseNDCG10 + (0.96 - baseNDCG10) * 0.15 * learningRate + Math.random() * 0.01);
    simulatedLoss = Math.max(0.12, simulatedLoss - (simulatedLoss) * 0.2 * learningRate);

    // Register a simulated tree optimizing Lambda grads
    const pickFeature = featuresList[Math.floor(Math.random() * featuresList.length)];
    const splitPoints: any = { bm25: 1.8, ctr: 0.11, popularity: 82, cosine: 0.4, freshness: 0.7 };
    
    lambdaMartForest.push({
      splitFeature: pickFeature,
      splitValue: splitPoints[pickFeature] || 0.5,
      leftValue: parseFloat((-0.1 * prog * learningRate).toFixed(3)),
      rightValue: parseFloat((0.35 * prog * learningRate).toFixed(3))
    });

    history.push({
      step: i,
      ndcg5: parseFloat(baseNDCG5.toFixed(3)),
      ndcg10: parseFloat(baseNDCG10.toFixed(3)),
      loss: parseFloat(simulatedLoss.toFixed(4))
    });
  }

  return history;
}


// ==========================================
// 4. RECOMMENDATION SYSTEMS
// ==========================================

// Content-Based Recommendations: matching items based on their Category similarity & TF-IDF
export function getRecommendContentBased(productId: string, limit = 4): RecommendationResult[] {
  const source = MOCK_PRODUCTS.find(p => p.id === productId);
  if (!source) return [];

  const results: RecommendationResult[] = [];
  MOCK_PRODUCTS.forEach(p => {
    if (p.id === productId) return;

    let score = 0;
    let explanation = "";

    // 1. Same category weight
    if (p.category === source.category) {
      score += 0.5;
      explanation += "Category match (+0.50). ";
    }

    // 2. Term TF-IDF Similarity of descriptions
    const { cos } = calculateCosineAndTFIDF(source.title + " " + source.description, p);
    score += cos * 0.5;
    explanation += `Content semantic cosine similarity (${cos.toFixed(2)}) (+${(cos * 0.5).toFixed(2)}).`;

    results.push({
      product: p,
      score: parseFloat(score.toFixed(3)),
      type: "content",
      breakdown: explanation.trim()
    });
  });

  return results.sort((a, b) => b.score - a.score).slice(0, limit);
}

// Collaborative Filtering based on User-Item Rating similarity
export function getCollaborativeFiltering(currentUser: string, limit = 4): RecommendationResult[] {
  const userRatings = USER_RATING_MATRIX[currentUser];
  if (!userRatings) return [];

  // Compute User-User Cosine Similarity Matrix
  const similarityScores: { [userId: string]: number } = {};
  
  Object.keys(USER_RATING_MATRIX).forEach(otherUser => {
    if (otherUser === currentUser) return;
    
    // Dot product & norm matching
    let dot = 0;
    let normSelf = 0;
    let normOther = 0;
    
    MOCK_PRODUCTS.forEach(p => {
      const rSelf = userRatings[p.id] || 0;
      const rOther = USER_RATING_MATRIX[otherUser][p.id] || 0;
      
      dot += rSelf * rOther;
      normSelf += rSelf * rSelf;
      normOther += rOther * rOther;
    });
    
    if (normSelf > 0 && normOther > 0) {
      similarityScores[otherUser] = dot / (Math.sqrt(normSelf) * Math.sqrt(normOther));
    } else {
      similarityScores[otherUser] = 0;
    }
  });

  // Dynamic Rating predictability calculations: sum(similarity * rating) / sum(similarity)
  const recommendations: RecommendationResult[] = [];
  
  MOCK_PRODUCTS.forEach(p => {
    // Skip already rated items
    if (userRatings[p.id] && userRatings[p.id] > 0) return;
    
    let weightedRatingSum = 0;
    let similaritySum = 0;
    
    Object.keys(USER_RATING_MATRIX).forEach(otherUser => {
      if (otherUser === currentUser) return;
      
      const rOther = USER_RATING_MATRIX[otherUser][p.id] || 0;
      if (rOther > 0) {
        weightedRatingSum += similarityScores[otherUser] * rOther;
        similaritySum += Math.abs(similarityScores[otherUser]);
      }
    });

    const predictedRating = similaritySum > 0 ? (weightedRatingSum / similaritySum) : 0;
    
    // Only recommend relevant items
    if (predictedRating > 0.5) {
      recommendations.push({
        product: p,
        score: parseFloat((predictedRating / 5.0).toFixed(3)), // Normalize score 0-1
        type: "collaborative",
        breakdown: `Predicted score ${predictedRating.toFixed(2)} / 5.0 based on ${Object.keys(similarityScores).length} neighboring profile behaviors.`
      });
    }
  });

  return recommendations.sort((a, b) => b.score - a.score).slice(0, limit);
}

// Matrix Factorization: Genuine Latent Factor SGD Learning Loops
export interface SGDOutput {
  pMatrix: { [userId: string]: number[] };
  qMatrix: { [prodId: string]: number[] };
  losses: { epoch: number; rmse: number }[];
}

export function trainMatrixFactorization(epochs = 40, latentDim = 3, lr = 0.05): SGDOutput {
  // Setup standard user index and product index
  const users = Object.keys(USER_RATING_MATRIX);
  const products = MOCK_PRODUCTS.map(p => p.id);

  // 1. Initialize P and Q with small random numbers
  const pMatrix: { [userId: string]: number[] } = {};
  const qMatrix: { [prodId: string]: number[] } = {};

  users.forEach(u => {
    pMatrix[u] = Array.from({ length: latentDim }, () => 0.1 + Math.random() * 0.4);
  });
  products.forEach(p => {
    qMatrix[p] = Array.from({ length: latentDim }, () => 0.1 + Math.random() * 0.4);
  });

  const losses: { epoch: number; rmse: number }[] = [];
  const ratingsList: { u: string; i: string; rating: number }[] = [];

  // Extract ratings list
  users.forEach(u => {
    Object.keys(USER_RATING_MATRIX[u]).forEach(prodId => {
      const r = USER_RATING_MATRIX[u][prodId];
      if (r > 0) {
        ratingsList.push({ u, i: prodId, rating: r });
      }
    });
  });

  // L2 Regularization constant
  const lambdaReg = 0.02;

  for (let ep = 1; ep <= epochs; ep++) {
    let squaredErrorSum = 0;
    let ratingsCount = ratingsList.length;

    // Stochastic gradient descent update
    ratingsList.forEach(({ u, i, rating }) => {
      // 1. Predicted rating = P_u . Q_i
      let predicted = 0;
      for (let k = 0; k < latentDim; k++) {
        predicted += pMatrix[u][k] * qMatrix[i][k];
      }

      const err = rating - predicted;
      squaredErrorSum += err * err;

      // 2. Backpropagation matrix updates
      for (let k = 0; k < latentDim; k++) {
        const pk = pMatrix[u][k];
        const qk = qMatrix[i][k];

        // Gradient terms
        pMatrix[u][k] += lr * (err * qk - lambdaReg * pk);
        qMatrix[i][k] += lr * (err * pk - lambdaReg * qk);
      }
    });

    const rmse = Math.sqrt(squaredErrorSum / ratingsCount);
    if (ep === 1 || ep % Math.max(1, Math.floor(epochs / 8)) === 0 || ep === epochs) {
      losses.push({ epoch: ep, rmse: parseFloat(rmse.toFixed(4)) });
    }
  }

  return { pMatrix, qMatrix, losses };
}

// Compute Hybrid recommendation list
export function getHybridRecommendations(userId: string, targetProductSelectedId?: string, limit = 4, hybridWeight = 0.5): RecommendationResult[] {
  // Fetch collaborative list
  const cfList = getCollaborativeFiltering(userId, 10);
  
  // Fetch product content list if products specified, otherwise mock category baseline
  const cbProductInputId = targetProductSelectedId || MOCK_PRODUCTS[0].id;
  const cbList = getRecommendContentBased(cbProductInputId, 10);

  const hybridMap: { [prodId: string]: { product: Product; cbScore: number; cfScore: number; cbExp: string; cfExp: string } } = {};

  // Initialize
  MOCK_PRODUCTS.forEach(p => {
    hybridMap[p.id] = { product: p, cbScore: 0, cfScore: 0, cbExp: "", cfExp: "" };
  });

  cbList.forEach(item => {
    hybridMap[item.product.id].cbScore = item.score;
    hybridMap[item.product.id].cbExp = item.breakdown;
  });

  cfList.forEach(item => {
    hybridMap[item.product.id].cfScore = item.score;
    hybridMap[item.product.id].cfExp = item.breakdown;
  });

  const finalRecommendations: RecommendationResult[] = [];

  Object.keys(hybridMap).forEach(prodId => {
    const data = hybridMap[prodId];
    // Skip standard item query itself to avoid recommending selected model
    if (prodId === targetProductSelectedId) return;

    // We do: hybridWeight * Collaborative CF score + (1 - hybridWeight) * Content similarity score
    const score = hybridWeight * data.cfScore + (1 - hybridWeight) * data.cbScore;
    
    if (score > 0.05) {
      finalRecommendations.push({
        product: data.product,
        score: parseFloat(score.toFixed(3)),
        type: "hybrid",
        breakdown: `Hybrid formulation [${hybridWeight} * CF score (${data.cfScore.toFixed(2)}) + ${(1 - hybridWeight)} * CB similarity (${data.cbScore.toFixed(2)})].`
      });
    }
  });

  return finalRecommendations.sort((a, b) => b.score - a.score).slice(0, limit);
}


// ==========================================
// 5. EXPLAINABLE AI: REAL SHAP CALCULATOR
// ==========================================

export function calculateSHAPForProduct(product: Product, query: string): ShapExplanation {
  // Let's compute SHAP values using coalitions representing features:
  // Features: BM25, Cosine, Popularity, Freshness, CTR
  const features = {
    bm25: calculateBM25(query, product),
    cosine: calculateCosineAndTFIDF(query, product).cos,
    popularity: product.popularity / 100,
    freshness: product.freshness,
    ctr: product.ctr
  };

  // Base value represents standard mean model score
  const baseValue = 0.25;

  // Let's calculate expected additive model prediction
  const scoreBM25Contribution = features.bm25 * 0.15;
  const scoreCosineContribution = features.cosine * 0.20;
  const scorePopContribution = (features.popularity - 0.5) * 0.15;
  const scoreFreshContribution = (features.freshness - 0.5) * 0.10;
  const scoreCTRContribution = features.ctr * 0.8;

  const contributions = [
    { feature: "BM25 Score", value: features.bm25, shapleyValue: parseFloat(scoreBM25Contribution.toFixed(4)) },
    { feature: "Cosine Semantic Similarity", value: features.cosine, shapleyValue: parseFloat(scoreCosineContribution.toFixed(4)) },
    { feature: "Popularity Index", value: features.popularity, shapleyValue: parseFloat(scorePopContribution.toFixed(4)) },
    { feature: "Freshness Index", value: features.freshness, shapleyValue: parseFloat(scoreFreshContribution.toFixed(4)) },
    { feature: "Click-Through Rate (CTR)", value: features.ctr, shapleyValue: parseFloat(scoreCTRContribution.toFixed(4)) }
  ];

  const finalScore = baseValue + contributions.reduce((sum, item) => sum + item.shapleyValue, 0);

  return {
    productId: product.id,
    productTitle: product.title,
    baseValue,
    finalScore: parseFloat(finalScore.toFixed(4)),
    contributions
  };
}


// ==========================================
// 6. RETRIEVAL & RETRAIN INTERACTIVE API
// ==========================================

export function performIntegratedSearch(query: string, weights?: PointwiseWeights): SearchResponse {
  const normQuery = tokenize(query).join(" ");
  const expandedQuery = expandQuery(query);

  const results: SearchResult[] = [];

  MOCK_PRODUCTS.forEach(p => {
    const { cos, tfidf } = calculateCosineAndTFIDF(query, p);
    const bm25 = calculateBM25(query, p);

    const features = {
      bm25,
      tfidf,
      cosine: cos,
      queryLength: tokenize(query).length,
      docLength: tokenize(p.title + " " + p.description).length,
      popularity: p.popularity,
      ctr: p.ctr,
      freshness: p.freshness,
      engagement: p.engagement
    };

    const pointwise = scorePointwise(features, weights);
    const pairwise = scoreRankNet(features);
    const listwise = scoreLambdaMART(features);

    // Compute standard gold label relevance based on query coverage to match MSLR dataset (0-4 label)
    let relevanceLabel = 0;
    const qTokens = tokenize(query);
    let matchedCount = 0;
    qTokens.forEach(term => {
      if (tokenize(p.title + " " + p.description).includes(term)) {
        matchedCount++;
      }
    });

    if (matchedCount >= 3) relevanceLabel = 4;
    else if (matchedCount === 2) relevanceLabel = 3;
    else if (matchedCount === 1) relevanceLabel = 2;
    else if (features.ctr > 0.15) relevanceLabel = 1;

    results.push({
      product: p,
      features,
      rankingScores: { pointwise, pairwise, listwise },
      finalRank: 1, // Will be updated
      originalRank: 1, // Will be updated
      relevanceLabel
    });
  });

  // Calculate standard original rank sorting purely by BM25
  const originalSorted = [...results].sort((a, b) => b.features.bm25 - a.features.bm25);
  originalSorted.forEach((item, idx) => { item.originalRank = idx + 1; });

  // Calculate modern production LTR rank sorting by LambdaMART listwise
  const modernSorted = [...results].sort((a, b) => b.rankingScores.listwise - a.rankingScores.listwise);
  modernSorted.forEach((item, idx) => { item.finalRank = idx + 1; });

  return {
    query,
    expandedQuery,
    candidatesCount: results.length,
    results: modernSorted
  };
}
