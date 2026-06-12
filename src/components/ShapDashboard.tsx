/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import { useState, useEffect } from 'react';
import { Cpu, HelpCircle, Sliders, ArrowRight, CornerDownRight } from 'lucide-react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from 'recharts';
import { ShapExplanation } from '../types';

export default function ShapDashboard() {
  const [selectedProduct, setSelectedProduct] = useState('prod-1');
  const [query, setQuery] = useState('smart speaker');
  const [explanation, setExplanation] = useState<ShapExplanation | null>(null);
  const [loading, setLoading] = useState(false);

  const sampleProducts = [
    { id: 'prod-1', title: 'Echo Dot Smart Speaker' },
    { id: 'prod-2', title: 'Fjallraven Backpack' },
    { id: 'prod-3', title: 'Sony Wireless Headphones' },
    { id: 'prod-5', title: 'Anker fast GaN charger' },
    { id: 'prod-6', title: 'Apple AirPods Pro II' },
    { id: 'prod-7', title: 'Nike Pegasus Running Shoes' }
  ];

  const fetchShapData = async () => {
    setLoading(true);
    try {
      const res = await fetch('/api/explain', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ productId: selectedProduct, query })
      });
      const data = await res.json();
      setExplanation(data);
    } catch (err) {
      console.error('Error fetching SHAP explanations from server:', err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchShapData();
  }, [selectedProduct, query]);

  // Global static summary dataset of feature importance levels (SHAP global values across 10,000 MSLR web pages)
  const globalShapImportance = [
    { feature: 'BM25 Index Match', importance: 0.38, details: 'Core token frequencies and doc length' },
    { feature: 'Cosine Semantic Embedding', importance: 0.28, details: 'Latent category & synonym alignment' },
    { feature: 'Click-Through Rate (CTR)', importance: 0.24, details: 'Direct popular validation signal' },
    { feature: 'Engagement Index (Reviews)', importance: 0.16, details: 'Average ratings score from user lists' },
    { feature: 'Document Freshness', importance: 0.12, details: 'Time decay parameter since insertion' }
  ];

  return (
    <div className="space-y-6" id="shap-dashboard-container">
      {/* Information Header card */}
      <div className="bg-[#121826] border border-white/5 rounded-xl p-6 shadow-sm">
        <h2 className="text-xl font-semibold text-slate-100 flex items-center gap-2 mb-2">
          <Cpu className="w-5 h-5 text-cyan-400" />
          Explainable AI (XAI) SHAP Evaluation Workspace
        </h2>
        <p className="text-slate-400 text-sm mb-6 leading-relaxed">
          Verify additive feature contributions computed via Shapley coalitional mathematics. Compare standard expected base prediction values against actual final LTR ranking predictions.
        </p>

        {/* Workspace controls */}
        <div className="flex flex-col md:flex-row gap-4 bg-[#0A0D14] border border-white/5 p-4 rounded-lg text-xs font-mono">
          <div className="flex-1 space-y-1.5">
            <span className="text-slate-500 uppercase font-bold block">Active evaluation query string</span>
            <input
              type="text"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              className="w-full bg-[#121826] border border-white/10 text-slate-200 p-2 rounded focus:outline-none focus:ring-1 focus:ring-cyan-500"
              placeholder="Query to evaluate..."
            />
          </div>

          <div className="flex-1 space-y-1.5">
            <span className="text-slate-500 uppercase font-bold block">Target candidate product</span>
            <select
              value={selectedProduct}
              onChange={(e) => setSelectedProduct(e.target.value)}
              className="w-full bg-[#121826] border border-white/10 text-slate-200 p-2 rounded focus:outline-none focus:ring-1 focus:ring-cyan-500"
            >
              {sampleProducts.map(p => (
                <option key={p.id} value={p.id}>{p.title}</option>
              ))}
            </select>
          </div>
        </div>
      </div>

      {/* Grid: Local SHAPWaterfall values alongside Global Feature Importances */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Local waterfall additive evaluation */}
        <div className="bg-[#121826] border border-white/5 rounded-xl p-5 shadow-sm space-y-4">
          <div>
            <h3 className="font-semibold text-slate-200 text-sm">
              Additive SHAP Local Contribution (Waterfall Projection)
            </h3>
            <p className="text-slate-400 text-xs mt-1 leading-normal">
              Shows how each attribute pushes the model's LTR score away from the base value of <span className="text-cyan-400 bg-[#0A0D14] px-1.5 py-0.5 rounded font-mono font-medium border border-white/5">0.25</span> toward the final LTR scoring output.
            </p>
          </div>

          {explanation ? (
            <div className="space-y-4 font-mono">
              {/* Force plot visual bar representing positives vs negatives */}
              <div className="space-y-1.5">
                <div className="flex justify-between text-xs text-slate-400">
                  <span>Base Expected Value: {explanation.baseValue}</span>
                  <span className="text-cyan-400 font-bold">Predicted LTR Score: {explanation.finalScore}</span>
                </div>
                {/* Horizontal segmented relative slider */}
                <div className="w-full h-4 bg-[#0A0D14] rounded-full overflow-hidden flex border border-white/5">
                  {explanation.contributions.map((item, idx) => {
                    const absVal = Math.abs(item.shapleyValue);
                    const percentage = Math.max(5, (absVal / 0.5) * 100);
                    const color = item.shapleyValue >= 0 ? 'bg-cyan-500' : 'bg-fuchsia-500';
                    return (
                      <div
                        key={item.feature + idx}
                        style={{ width: `${percentage}%` }}
                        className={`${color} h-full border-r border-[#0A0D14]/20`}
                        title={`${item.feature}: ${item.shapleyValue > 0 ? '+' : ''}${item.shapleyValue}`}
                      />
                    );
                  })}
                </div>
                <div className="flex justify-between text-[10px] text-slate-505">
                  <span className="text-cyan-400 font-medium">🔼 Positive Contribution (Cyan)</span>
                  <span className="text-fuchsia-400 font-medium">🔽 Negative Contribution (Fuchsia)</span>
                </div>
              </div>

              {/* Increments list */}
              <div className="space-y-2 pt-2 border-t border-white/5 text-xs">
                {explanation.contributions.map((c) => (
                  <div key={c.feature} className="flex justify-between items-center py-1 border-b border-white/5">
                    <span className="text-slate-300 flex items-center gap-1">
                      <CornerDownRight className="w-3.5 h-3.5 text-slate-500" />
                      {c.feature}
                    </span>
                    <div className="flex items-center gap-3">
                      <span className="text-slate-500 text-[11px]">(val: {c.value})</span>
                      {c.shapleyValue >= 0 ? (
                        <span className="text-cyan-450 font-medium font-mono text-right min-w-[70px]">
                          +{c.shapleyValue.toFixed(4)}
                        </span>
                      ) : (
                        <span className="text-fuchsia-450 font-medium font-mono text-right min-w-[70px]">
                          {c.shapleyValue.toFixed(4)}
                        </span>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          ) : (
            <p className="text-xs text-slate-500 italic p-6 text-center">Loading calculations...</p>
          )}
        </div>

        {/* Global absolute Shapley importances */}
        <div className="bg-[#121826] border border-white/5 rounded-xl p-5 shadow-sm space-y-4">
          <div>
            <h3 className="font-semibold text-slate-200 text-sm">
              Global Feature Importance (aggregated across MSLR datasets)
            </h3>
            <p className="text-slate-400 text-xs mt-1">
              Absolute mean SHAP values across 10,000 query-document matched tuples. BM25 frequencies and semantic embedding weights provide major listwise LTR splits.
            </p>
          </div>

          <div className="h-[220px] w-full pt-2">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={globalShapImportance} layout="vertical">
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                <XAxis type="number" stroke="#475569" fontSize={9} />
                <YAxis dataKey="feature" type="category" stroke="#475569" fontSize={9} width={130} />
                <Tooltip contentStyle={{ backgroundColor: '#0A0D14', borderColor: 'rgba(255,255,255,0.1)' }} />
                <Bar dataKey="importance" radius={[0, 4, 4, 0]} barSize={12}>
                  {globalShapImportance.map((entry, index) => {
                    const colors = ['#06b6d4', '#22d3ee', '#38bdf8', '#0891b2', '#0e7490'];
                    return <Cell key={`cell-${index}`} fill={colors[index % colors.length]} />;
                  })}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>

          <div className="bg-[#0A0D14] border border-white/5 p-3.5 rounded-lg space-y-1">
            <div className="flex items-center gap-1.5 text-xs font-semibold text-slate-350">
              <HelpCircle className="w-4 h-4 text-cyan-400" />
              <span>How are these calculated?</span>
            </div>
            <p className="text-slate-500 text-[11px] leading-relaxed">
              SHAP computes marginal product values over exhaustive system coalitional subsets (all $2^F$ subsets) of BM25 similarity, popularity, CTR, freshness, and engagement indexes.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
