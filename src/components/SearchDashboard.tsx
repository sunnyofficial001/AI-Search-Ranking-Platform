/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import { useState } from 'react';
import { Search, Sparkles, TrendingUp, SlidersHorizontal, ArrowRight, CornerDownRight } from 'lucide-react';
import { SearchResult } from '../types';

export default function SearchDashboard() {
  const [query, setQuery] = useState('smart speaker');
  const [expandedQuery, setExpandedQuery] = useState('');
  const [useGemini, setUseGemini] = useState(false);
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState<SearchResult[]>([]);
  const [aiExplanation, setAiExplanation] = useState('');

  const [activeWeights, setActiveWeights] = useState({
    bm25: 0.35,
    tfidf: 0.15,
    cosine: 0.10,
    ctr: 0.12,
    freshness: 0.08,
    popularity: 0.10,
    engagement: 0.10
  });

  const handleSearch = async () => {
    setLoading(true);
    try {
      // 1. Get query expansion from Gemini or fallback
      let expQuery = '';
      if (useGemini) {
        const gemRes = await fetch('/api/gemini/expand', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ query, mode: 'search_intent' })
        });
        const gemData = await gemRes.json();
        expQuery = gemData.expanded || query;
        setAiExplanation(gemData.explanation || 'Semantic synonyms generated to bridge the gap.');
      } else {
        setAiExplanation('Rule-based query expansion used to increase candidate recall.');
      }

      // 2. Perform server retrieval
      const res = await fetch('/api/search', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: expQuery || query, weights: activeWeights })
      });
      const data = await res.json();
      setExpandedQuery(data.expandedQuery || query);
      setResults(data.results || []);
    } catch (err) {
      console.error('Error during search fetch:', err);
    } finally {
      setLoading(false);
    }
  };

  const preloadSampleQuery = (q: string) => {
    setQuery(q);
  };

  const handleWeightChange = (key: string, val: number) => {
    setActiveWeights(prev => ({ ...prev, [key]: parseFloat(val.toFixed(2)) }));
  };

  return (
    <div className="space-y-6" id="search-dashboard-container">
      {/* Search Header and Inputs */}
      <div className="bg-[#121826] border border-white/5 rounded-xl p-6 shadow-sm">
        <h2 className="text-xl font-semibold text-slate-100 flex items-center gap-2 mb-2">
          <Search className="w-5 h-5 text-cyan-400" />
          Elasticsearch BM25 Candidate Generation & LTR Ranking
        </h2>
        <p className="text-slate-400 text-sm mb-6 leading-relaxed">
          Input clean search phrases. Toggle AI semantic expansion to see how embedding vectors and synoymystic attributes enhance BM25 scoring.
        </p>

        <div className="flex flex-col md:flex-row gap-3">
          <div className="relative flex-1">
            <span className="absolute inset-y-0 left-0 flex items-center pl-3">
              <Search className="w-5 h-5 text-slate-500" />
            </span>
            <input
              type="text"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="e.g. noise cancelling headphones, fast charger..."
              className="w-full bg-[#0A0D14] border border-white/10 text-slate-100 pl-10 pr-4 py-3 rounded-lg focus:outline-none focus:ring-2 focus:ring-cyan-500 text-sm transition-all"
              id="search-input"
            />
          </div>
          <button
            onClick={handleSearch}
            disabled={loading}
            className="bg-cyan-500 hover:bg-cyan-455 text-slate-950 font-bold text-sm px-6 py-3 rounded-lg transition-colors flex items-center justify-center gap-2 disabled:opacity-50"
            id="search-submit-btn"
          >
            {loading ? 'Retrieving...' : 'Search'}
            <ArrowRight className="w-4 h-4" />
          </button>
        </div>

        {/* Query expansion control settings */}
        <div className="mt-4 pt-4 border-t border-white/5 flex flex-wrap items-center justify-between gap-4">
          <div className="flex items-center gap-6">
            <label className="flex items-center gap-2 cursor-pointer text-xs text-slate-300 font-medium">
              <input
                type="checkbox"
                checked={useGemini}
                onChange={(e) => setUseGemini(e.target.checked)}
                className="rounded border-white/10 bg-[#0A0D14] text-cyan-500 focus:ring-cyan-500 h-4 w-4"
              />
              <Sparkles className="w-3.5 h-3.5 text-fuchsia-400 inline" />
              Enable Gemini Server-side Query Expansion Model
            </label>
          </div>

          <div className="flex gap-2 text-xs">
            <span className="text-slate-500 self-center">Try queries:</span>
            {['smart speaker', 'headphones', 'backpack', 'charger'].map(term => (
              <button
                key={term}
                onClick={() => preloadSampleQuery(term)}
                className="bg-[#0A0D14] hover:bg-white/5 border border-white/5 text-slate-300 px-2.5 py-1 rounded transition-colors"
              >
                "{term}"
              </button>
            ))}
          </div>
        </div>

        {/* Expansion Response */}
        {expandedQuery && (
          <div className="mt-4 p-4 bg-[#0A0D14] rounded-lg border border-white/5 text-xs text-slate-300 space-y-2 animate-fade-in">
            <div className="flex items-center gap-2 text-cyan-400 font-mono">
              <CornerDownRight className="w-4 h-4" />
              <span>Expanded Lucene Query:</span>
            </div>
            <p className="bg-[#121826] px-3 py-2 border border-white/5 text-slate-100 rounded font-mono select-all">
              {expandedQuery}
            </p>
            {aiExplanation && (
              <p className="text-slate-400 italic">
                💡 <span className="font-semibold text-slate-300">Synonym Brief:</span> {aiExplanation}
              </p>
            )}
          </div>
        )}
      </div>

      {/* Grid: Tuning weights alongside output lists */}
      <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
        {/* Tuning weights configuration */}
        <div className="lg:col-span-1 bg-[#121826] border border-white/5 rounded-xl p-5 shadow-sm space-y-4 h-fit">
          <h3 className="font-semibold text-slate-200 text-sm flex items-center gap-2 border-b border-white/5 pb-3">
            <SlidersHorizontal className="w-4 h-4 text-cyan-400" />
            Static Pointwise Weights
          </h3>
          <p className="text-slate-400 text-xs leading-relaxed">
            Configure linear pointwise score calculations. Changes affect how standard pointwise score is evaluated.
          </p>

          <div className="space-y-4">
            {Object.keys(activeWeights).map((key) => (
              <div key={key} className="space-y-1">
                <div className="flex justify-between text-xs font-mono">
                  <span className="text-slate-400 uppercase">{key}</span>
                  <span className="text-slate-200">{(activeWeights as any)[key]}</span>
                </div>
                <input
                  type="range"
                  min="0"
                  max="1"
                  step="0.05"
                  value={(activeWeights as any)[key]}
                  onChange={(e) => handleWeightChange(key, parseFloat(e.target.value))}
                  className="w-full accent-cyan-500 h-1 bg-[#0A0D14] rounded-lg cursor-pointer"
                />
              </div>
            ))}
          </div>
        </div>

        {/* Candidates output search results compare */}
        <div className="lg:col-span-3 space-y-4">
          <div className="flex items-center justify-between">
            <h3 className="text-sm font-semibold text-slate-300">
              Retrieved Candidates List ({results.length} items found)
            </h3>
            <span className="text-xs bg-[#121826] text-cyan-400 border border-white/5 px-2.5 py-1 rounded-full font-mono">
              Production Sort: LambdaMART LTR
            </span>
          </div>

          {results.length === 0 ? (
            <div className="bg-[#121826] border border-white/5 rounded-xl p-12 text-center text-slate-500">
              <p className="text-sm mb-2">No query is active.</p>
              <p className="text-xs text-slate-600">Please click the search button above or select a sample query to retrieve simulated index files.</p>
            </div>
          ) : (
            <div className="space-y-3">
              {results.map((item, index) => {
                const rankDelta = item.originalRank - item.finalRank;
                return (
                  <div
                    key={item.product.id}
                    className="bg-[#121826] border border-white/5 rounded-xl p-5 hover:border-white/10 transition-all flex flex-col md:flex-row justify-between gap-4"
                    id={`search-result-${item.product.id}`}
                  >
                    <div className="flex-1 space-y-2">
                      <div className="flex items-start gap-2 flex-wrap">
                        <span className="bg-[#0A0D14] text-cyan-400 text-[10px] font-mono font-medium px-2 py-0.5 rounded border border-white/5">
                          {item.product.category}
                        </span>
                        <span className="bg-[#0A0D14] text-emerald-400 text-[10px] font-mono font-medium px-2 py-0.5 rounded border border-white/5">
                          ID: {item.product.id}
                        </span>
                        <span className="bg-[#0A0D14] text-amber-450 text-[10px] font-mono font-medium px-2 py-0.5 rounded border border-white/5">
                          Label Score: {item.relevanceLabel}/4
                        </span>
                      </div>

                      <h4 className="text-base font-semibold text-slate-100 leading-tight">
                        {item.product.title}
                      </h4>
                      <p className="text-slate-400 text-xs line-clamp-2 leading-relaxed">
                        {item.product.description}
                      </p>

                      {/* Score metrics blocks */}
                      <div className="pt-2 flex flex-wrap gap-4 text-[10px] font-mono text-slate-400">
                        <div className="bg-[#0A0D14] px-2.5 py-1 rounded border border-white/5">
                          BM25: <span className="text-slate-200">{item.features.bm25}</span>
                        </div>
                        <div className="bg-[#0A0D14] px-2.5 py-1 rounded border border-white/5">
                          Cosine Semantic: <span className="text-slate-200">{item.features.cosine}</span>
                        </div>
                        <div className="bg-[#0A0D14] px-2.5 py-1 rounded border border-white/5">
                          CTR: <span className="text-slate-200">{item.features.ctr}%</span>
                        </div>
                        <div className="bg-[#0A0D14] px-2.5 py-1 rounded border border-white/5">
                          Freshness: <span className="text-slate-200">{item.features.freshness}</span>
                        </div>
                        <div className="bg-[#0A0D14] px-2.5 py-1 rounded border border-white/5">
                          Engagement: <span className="text-slate-200">{item.features.engagement}⭐</span>
                        </div>
                      </div>
                    </div>

                    {/* Rank indices compares */}
                    <div className="flex md:flex-col justify-between md:justify-center items-center md:items-end gap-3 border-t md:border-t-0 border-white/5 pt-3 md:pt-0 min-w-[120px]">
                      <div className="flex md:flex-col gap-2 items-center md:items-end">
                        <div className="text-right">
                          <p className="text-[10px] text-slate-500 uppercase font-mono">LTR Revised Rank</p>
                          <p className="text-2xl font-black text-cyan-400 font-mono">#{item.finalRank}</p>
                        </div>
                        <div className="text-right">
                          <p className="text-[10px] text-slate-500 uppercase font-mono">BM25 Rank</p>
                          <p className="text-sm font-semibold text-slate-400 font-mono">#{item.originalRank}</p>
                        </div>
                      </div>

                      {/* Rank promotion arrow indicator */}
                      <div>
                        {rankDelta > 0 ? (
                          <span className="text-emerald-400 text-xs px-2 py-1 rounded bg-emerald-950/20 border border-emerald-900/40 flex items-center gap-1 font-mono">
                            ▲ +{rankDelta} Promoted
                          </span>
                        ) : rankDelta < 0 ? (
                          <span className="text-fuchsia-400 text-xs px-2 py-1 rounded bg-fuchsia-950/20 border border-fuchsia-900/40 flex items-center gap-1 font-mono">
                            ▼ {rankDelta} Demoted
                          </span>
                        ) : (
                          <span className="text-slate-500 text-xs px-2 py-1 rounded bg-[#0A0D14] border border-white/5 flex items-center gap-1 font-mono">
                            ➖ Neutral
                          </span>
                        )}
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
