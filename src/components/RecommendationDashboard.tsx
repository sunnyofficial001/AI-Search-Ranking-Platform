/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import { useState } from 'react';
import { Compass, Users, Sliders, Play, TrendingDown, Eye, CheckCircle, Smartphone } from 'lucide-react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import { RecommendationResult } from '../types';

export default function RecommendationDashboard() {
  const [activeUser, setActiveUser] = useState('user-1');
  const [activeProduct, setActiveProduct] = useState('prod-1');
  const [recType, setRecType] = useState<'content' | 'collaborative' | 'matrix_factorization' | 'hybrid'>('hybrid');
  const [hybridWeight, setHybridWeight] = useState(0.5);

  // Matrix factorization parameters
  const [epochs, setEpochs] = useState(40);
  const [latentDim, setLatentDim] = useState(3);
  const [learningRate, setLearningRate] = useState(0.05);

  const [loading, setLoading] = useState(false);
  const [recResults, setRecResults] = useState<RecommendationResult[]>([]);
  const [mfLosses, setMfLosses] = useState<any[]>([]);

  // Sample representations matching USER_RATING_MATRIX keys
  const sampleUsers = ['user-1', 'user-2', 'user-3', 'user-4', 'user-5'];

  // All sample products in indexing to use as target Content-based keys
  const productsList = [
    { id: 'prod-1', name: 'Echo Dot Smart Speaker' },
    { id: 'prod-2', name: 'Fjallraven Backpack' },
    { id: 'prod-3', name: 'Sony Wireless Headphones' },
    { id: 'prod-5', name: 'Anker fast GaN charger' },
    { id: 'prod-6', name: 'Apple AirPods Pro II' },
    { id: 'prod-7', name: 'Nike Pegasus Running Shoes' }
  ];

  const handleComputeRecommendations = async () => {
    setLoading(true);
    try {
      const body: any = {
        userId: activeUser,
        productId: activeProduct,
        type: recType,
        hybridWeight
      };

      if (recType === 'matrix_factorization') {
        body.epochs = epochs;
        body.latentDim = latentDim;
        body.lr = learningRate;
      }

      const res = await fetch('/api/recommend', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body)
      });

      const data = await res.json();
      setRecResults(data.results || []);

      if (data.losses) {
        setMfLosses(data.losses);
      } else {
        setMfLosses([]);
      }
    } catch (err) {
      console.error('Error fetching recommendations from platform:', err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-6" id="recommendation-dashboard-container">
      {/* Platform Summary Panel */}
      <div className="bg-[#121826] border border-white/5 rounded-xl p-6 shadow-sm">
        <h2 className="text-xl font-semibold text-slate-100 flex items-center gap-2 mb-2">
          <Compass className="w-5 h-5 text-cyan-400" />
          Multi-paradigm Recommendation Engine Orchestrator
        </h2>
        <p className="text-slate-400 text-sm mb-6 leading-relaxed">
          Leverage four state-of-the-art architectures: Content-Based, user Collaborative similarity matrices, latent Stochastic Gradient Descent (SGD) Matrix Factorization, or full structural Hybrid mixes.
        </p>

        {/* Global Parameter sliders and select forms */}
        <div className="grid grid-cols-1 lg:grid-cols-4 gap-6 bg-[#0A0D14] border border-white/5 p-5 rounded-lg text-xs">
          {/* Active profile selector */}
          <div className="space-y-1.5">
            <label className="text-slate-450 uppercase font-bold font-mono">User Session Persona</label>
            <select
              value={activeUser}
              onChange={(e) => setActiveUser(e.target.value)}
              className="w-full bg-[#121826] border border-white/10 text-slate-200 p-2.5 rounded focus:outline-none focus:ring-1 focus:ring-cyan-500 font-mono"
            >
              {sampleUsers.map(u => (
                <option key={u} value={u}>{u.toUpperCase()} (Profile rating cluster)</option>
              ))}
            </select>
          </div>

          {/* Core Rec strategy paradigm */}
          <div className="space-y-1.5">
            <label className="text-slate-450 uppercase font-bold font-mono">Recommendation Engine Paradigm</label>
            <select
              value={recType}
              onChange={(e: any) => setRecType(e.target.value)}
              className="w-full bg-[#121826] border border-white/10 text-slate-250 p-2.5 rounded focus:outline-none focus:ring-1 focus:ring-cyan-500 font-mono"
            >
              <option value="hybrid">Weighted Hybrid recommendation (CB + CF)</option>
              <option value="matrix_factorization">Latent Matrix Factorization SGD model</option>
              <option value="collaborative">User-User Collaborative Similarity</option>
              <option value="content flex">Content-Based tfidf item vector similarity</option>
            </select>
          </div>

          {/* Conditional setting: Content products or hybrid weights slider */}
          {recType === 'content' || recType === 'hybrid' ? (
            <div className="space-y-1.5">
              <label className="text-slate-450 uppercase font-bold font-mono">Target similarity product seed</label>
              <select
                value={activeProduct}
                onChange={(e) => setActiveProduct(e.target.value)}
                className="w-full bg-[#121826] border border-white/10 text-slate-250 p-2.5 rounded focus:outline-none"
              >
                {productsList.map(item => (
                  <option key={item.id} value={item.id}>{item.name}</option>
                ))}
              </select>
            </div>
          ) : recType === 'matrix_factorization' ? (
            <div className="space-y-1.5">
              <div className="flex justify-between text-slate-450 font-mono">
                <span className="uppercase font-bold">Latent dimensions (factors)</span>
                <span>k = {latentDim}</span>
              </div>
              <input
                type="range"
                min="2"
                max="5"
                step="1"
                value={latentDim}
                onChange={(e) => setLatentDim(parseInt(e.target.value))}
                className="w-full h-1 bg-[#121826] accent-cyan-500 rounded-lg cursor-pointer"
              />
            </div>
          ) : (
            <div className="text-slate-500 italic p-3 text-center border border-white/5 rounded bg-[#121826]/40">
              Collaborative utilizes User rating proximity metrics.
            </div>
          )}

          {/* Dynamic weighting or Training iterations */}
          {recType === 'hybrid' ? (
            <div className="space-y-1.5">
              <div className="flex justify-between text-slate-450 font-mono">
                <span className="uppercase font-bold">Hybrid Blend Ratio</span>
                <span>{Math.round(hybridWeight * 100)}% CF / {Math.round((1 - hybridWeight) * 100)}% CB</span>
              </div>
              <input
                type="range"
                min="0"
                max="1"
                step="0.05"
                value={hybridWeight}
                onChange={(e) => setHybridWeight(parseFloat(e.target.value))}
                className="w-full h-1 bg-[#121826] accent-fuchsia-500 rounded-lg cursor-pointer"
              />
            </div>
          ) : recType === 'matrix_factorization' ? (
            <div className="space-y-1.5">
              <div className="flex justify-between text-slate-450 font-mono">
                <span className="uppercase font-bold">SGD learning rate & epochs</span>
                <span>{learningRate} lr, {epochs} epochs</span>
              </div>
              <div className="flex gap-2 text-slate-200">
                <input
                  type="number"
                  min="0.01"
                  max="0.2"
                  step="0.01"
                  value={learningRate}
                  onChange={(e) => setLearningRate(parseFloat(e.target.value))}
                  className="w-1/2 bg-[#121826] border border-white/10 p-1.5 rounded text-center"
                  placeholder="LR"
                />
                <input
                  type="number"
                  min="10"
                  max="100"
                  step="5"
                  value={epochs}
                  onChange={(e) => setEpochs(parseInt(e.target.value))}
                  className="w-1/2 bg-[#121826] border border-white/10 p-1.5 rounded text-center"
                  placeholder="Epochs"
                />
              </div>
            </div>
          ) : (
            <div className="space-y-1">
              <span className="text-slate-500 select-none block">System configuration status</span>
              <span className="text-emerald-400 font-mono font-semibold">● Engine online</span>
            </div>
          )}
        </div>

        <div className="mt-4 flex justify-end">
          <button
            onClick={handleComputeRecommendations}
            disabled={loading}
            className="bg-cyan-500 hover:bg-cyan-400 text-slate-950 text-xs px-5 py-2.5 rounded-lg transition-all font-bold flex items-center gap-1.5 disabled:opacity-50"
            id="compute-recs-btn"
          >
            <Play className="w-3.5 h-3.5 fill-slate-950" />
            {loading ? 'Performing SGD Matrix factorization...' : 'Generate Catalog Recommendations'}
          </button>
        </div>
      </div>

      {/* Grid displays: Factorization SGD outputs alongside computed results */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Latent matrix factor learning curve (Recharts) */}
        {recType === 'matrix_factorization' && mfLosses.length > 0 && (
          <div className="lg:col-span-1 bg-[#121826] border border-white/5 rounded-xl p-5 shadow-sm space-y-4 h-fit animate-fade-in">
            <h3 className="font-semibold text-slate-200 text-sm flex items-center gap-2">
              <TrendingDown className="w-4 h-4 text-fuchsia-400" />
              SGD De-noising RMSE Learning Curve
            </h3>
            <p className="text-slate-400 text-xs leading-relaxed">
              Stochastic Gradient descent backpropagating errors across factors. Decreasing Root-Mean-Square-Error indicates model learns preferences matrix. This is executed natively on the server.
            </p>

            <div className="h-[180px] w-full pt-2">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={mfLosses}>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                  <XAxis dataKey="epoch" stroke="#475569" fontSize={9} />
                  <YAxis stroke="#475569" fontSize={9} />
                  <Tooltip contentStyle={{ backgroundColor: '#0A0D14', borderColor: 'rgba(255,255,255,0.1)' }} />
                  <Line type="monotone" dataKey="rmse" stroke="#f43f5e" name="RMSE error" strokeWidth={2.5} />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>
        )}

        {/* Results cards */}
        <div className={`${recType === 'matrix_factorization' && mfLosses.length > 0 ? 'lg:col-span-2' : 'lg:col-span-3'} space-y-4`}>
          <h3 className="text-sm font-semibold text-slate-350">
            Intelligent Recommended Catalog
          </h3>

          {recResults.length === 0 ? (
            <div className="bg-[#121826] border border-white/5 rounded-xl p-12 text-center text-slate-500">
              <p className="text-sm">No items loaded.</p>
              <p className="text-xs text-slate-600 mt-1">Please select parameters and click the catalog button above to invoke simulated prediction paths.</p>
            </div>
          ) : (
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {recResults.map((item) => (
                <div
                  key={item.product.id}
                  className="bg-[#121826] border border-white/5 p-5 rounded-xl hover:border-white/10 transition-all flex flex-col justify-between space-y-4"
                  id={`rec-item-${item.product.id}`}
                >
                  <div className="space-y-2">
                    <div className="flex justify-between items-start gap-2">
                      <span className="bg-[#0A0D14] text-cyan-400 border border-white/5 text-[10px] uppercase font-mono px-2 py-0.5 rounded font-bold">
                        {item.product.category}
                      </span>
                      <div className="text-right">
                        <span className="text-[10px] text-slate-500 uppercase font-mono block">Prediction Score</span>
                        <span className="text-sm font-black text-amber-450 font-mono">
                          {Math.round(item.score * 100)}% Match
                        </span>
                      </div>
                    </div>

                    <h4 className="text-sm font-bold text-slate-100 line-clamp-1">
                      {item.product.title}
                    </h4>
                    <p className="text-xs text-slate-400 line-clamp-2 leading-relaxed">
                      {item.product.description}
                    </p>
                  </div>

                  {/* Similarity breakdown detail */}
                  <div className="border-t border-white/5 pt-3 text-[11px] text-slate-500 italic space-y-1 leading-normal">
                    <div className="text-cyan-400 font-sans font-medium flex items-center gap-1">
                      <Eye className="w-3.5 h-3.5" />
                      <span>Mathematical Breakdown:</span>
                    </div>
                    <p>{item.breakdown}</p>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
