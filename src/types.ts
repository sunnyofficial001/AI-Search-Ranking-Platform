/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

export interface Product {
  id: string;
  title: string;
  description: string;
  category: string;
  popularity: number; // 1-100
  ctr: number; // Click-Through Rate (e.g. 0.01 - 0.45)
  freshness: number; // Score based on age (0-1)
  engagement: number; // Average review rating / duration (e.g. 1.0 - 5.0)
  createdAt: string;
}

export interface SearchResult {
  product: Product;
  features: {
    bm25: number;
    tfidf: number;
    cosine: number;
    queryLength: number;
    docLength: number;
    popularity: number;
    ctr: number;
    freshness: number;
    engagement: number;
  };
  rankingScores: {
    pointwise: number;
    pairwise: number;
    listwise: number; // LambdaMART
  };
  finalRank: number;
  originalRank: number;
  relevanceLabel: number; // 0-4 relevance score based on MSLR-WEB10K labels
}

export interface RecommendationResult {
  product: Product;
  score: number;
  type: 'content' | 'collaborative' | 'matrix_factorization' | 'hybrid';
  breakdown: string;
}

export interface ShapExplanation {
  productId: string;
  productTitle: string;
  baseValue: number;
  finalScore: number;
  contributions: {
    feature: string;
    value: number;
    shapleyValue: number;
  }[];
}

export interface ExperimentRun {
  runId: string;
  name: string;
  timestamp: string;
  algorithm: 'pointwise' | 'pairwise_ranknet' | 'listwise_lambdamart' | 'matrix_factorization';
  parameters: {
    [key: string]: string | number;
  };
  metrics: {
    ndcg5: number;
    ndcg10: number;
    map: number;
    mrr: number;
    precision5: number;
    recall5: number;
    [key: string]: number;
  };
  status: 'SUCCESS' | 'FAILED' | 'RUNNING';
}

export interface FeatureStoreRecord {
  productId: string;
  title: string;
  category: string;
  features: {
    bm25: number;
    tfidf: number;
    cosine: number;
    popularity: number;
    ctr: number;
    freshness: number;
    engagement: number;
  };
}

export interface SearchResponse {
  query: string;
  expandedQuery: string;
  candidatesCount: number;
  results: SearchResult[];
}
