/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import { useState } from 'react';
import { Database, Download, RefreshCw, CheckCircle, Sliders, Play, Info } from 'lucide-react';
import { FeatureStoreRecord } from '../types';

export default function DatasetFeatures() {
  const [pipelineState, setPipelineState] = useState<'idle' | 'downloading' | 'processing' | 'done'>('idle');
  const [pipelineProgress, setPipelineProgress] = useState(0);
  const [pipelineLogs, setPipelineLogs] = useState<string[]>([]);
  const [storeRecords, setStoreRecords] = useState<FeatureStoreRecord[]>([]);

  const startPipelineSimulator = () => {
    setPipelineState('downloading');
    setPipelineProgress(10);
    setPipelineLogs([
      '[INFO] Initiating MSLR-WEB10K dataset query preprocessing pipeline.',
      '[INGESTION] Downloading raw partition (Fold 1: train.txt)...',
      '[INGESTION] Connected to Microsoft Research repository: downloading 1.2MB compressed header configurations...'
    ]);

    // Stage 1: Ingestion simulation timeout (1.5 seconds)
    setTimeout(() => {
      setPipelineState('processing');
      setPipelineProgress(50);
      setPipelineLogs(prev => [
        ...prev,
        '[INGESTION] Download complete. 10,000 document matching listings parsed.',
        '[PIPELINE] Converting SVMLight query relevance labels (0 to 4)...',
        '[FEATURE ENGINEERING] Initializing TF-IDF vector matrices across vocab schemas...',
        '[FEATURE ENGINEERING] Formulating BM25 indexes with parameter metrics: k1 = 1.2, b = 0.75...',
        '[FEATURE ENGINEERING] Multiplying Cosine Similarity profiles on categories & reviews...',
        '[FEATURE_STORE] Registering engineered vectors inside centralized feature cache...'
      ]);
    }, 1500);

    // Stage 2: Processing committed timeout (3.4 seconds)
    setTimeout(async () => {
      setPipelineState('done');
      setPipelineProgress(100);
      setPipelineLogs(prev => [
        ...prev,
        '[SUCCESS] Preprocessing completed cleanly.',
        '[SUCCESS] Committed 10 production-quality featured profiles to the centralized Feature Store PostgreSQL simulated registry cache.'
      ]);

      // Retrieve Feature Store data from server DB simulation
      try {
        const res = await fetch('/api/feature-store');
        const data = await res.json();
        setStoreRecords(data);
      } catch (err) {
        console.error('Error fetching simulated Feature Store details:', err);
      }
    }, 3200);
  };

  return (
    <div className="space-y-6" id="dataset-features-container">
      {/* Information Header text */}
      <div className="bg-[#121826] border border-white/5 rounded-xl p-6 shadow-sm">
        <h2 className="text-xl font-semibold text-slate-100 flex items-center gap-2 mb-2">
          <Database className="w-5 h-5 text-cyan-400" />
          MSLR-WEB10K Preprocessing & Advanced Feature Store Registry
        </h2>
        <p className="text-slate-400 text-sm mb-6 leading-relaxed">
          Simulate downloading Microsoft Research Search Ranking sets (MSLR-WEB10K). Execute advanced feature engineering loops (BM25, TF-IDF, Cosine Similarity, CTR decays, Freshness profiles) and commit them into a relational Feature Store database.
        </p>

        {/* Dataset Metadata brief */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 border-t border-white/5 pt-4">
          <div className="flex items-start gap-2.5">
            <Info className="w-5 h-5 text-cyan-400 mt-0.5 shrink-0" />
            <div>
              <p className="text-xs font-semibold text-slate-200">MSLR-WEB10K Subset</p>
              <p className="text-slate-505 text-[11px] mt-0.5 leading-normal">
                Features 10,000 parsed queries, featuring 136 standard dimensional parameters (IDF match, coverages, CTR ratios).
              </p>
            </div>
          </div>
          <div className="flex items-start gap-2.5">
            <Download className="w-5 h-5 text-cyan-400 mt-0.5 shrink-0" />
            <div>
              <p className="text-xs font-semibold text-slate-200">ETL Parsing Engine</p>
              <p className="text-slate-505 text-[11px] mt-0.5 leading-normal">
                Maps SVMLight formats with query-URL relevance structures directly to feed Learning-to-Rank regression trees.
              </p>
            </div>
          </div>
          <div className="flex items-start gap-2.5">
            <Sliders className="w-5 h-5 text-cyan-400 mt-0.5 shrink-0" />
            <div>
              <p className="text-xs font-semibold text-slate-200">Real-time Feature Store</p>
              <p className="text-slate-505 text-[11px] mt-0.5 leading-normal">
                Maintains feature histories to avoid calculating expensive text similarity metrics (BM25, Cosine) on active search request paths.
              </p>
            </div>
          </div>
        </div>
      </div>

      {/* Grid: Trigger widget alongside store output catalog */}
      <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
        {/* ETL Pipeline controller */}
        <div className="lg:col-span-1 bg-[#121826] border border-white/5 rounded-xl p-5 shadow-sm space-y-4">
          <h3 className="font-semibold text-slate-200 text-sm">
            Trigger Preprocessing Pipeline
          </h3>
          <p className="text-slate-400 text-xs leading-normal">
            Initiate automated download and feature transformations workflows to register data columns.
          </p>

          <button
            onClick={startPipelineSimulator}
            disabled={pipelineState === 'downloading' || pipelineState === 'processing'}
            className="w-full bg-cyan-500 hover:bg-cyan-600 text-[#0A0D14] font-bold text-xs py-2.5 rounded transition-all flex items-center justify-center gap-1.5 disabled:opacity-50 cursor-pointer shadow-[0_0_12px_rgba(6,182,212,0.15)] uppercase"
            id="start-pipeline-btn"
          >
            <RefreshCw className={`w-3.5 h-3.5 ${pipelineState === 'downloading' || pipelineState === 'processing' ? 'animate-spin' : ''}`} />
            {pipelineState === 'idle' ? 'Invoke ETL Ingest' : pipelineState === 'done' ? 'Re-run ETL Ingest' : 'Running ETL...'}
          </button>

          {pipelineState !== 'idle' && (
            <div className="space-y-2 animate-fade-in pt-2">
              <div className="flex justify-between text-[11px] font-mono text-slate-400">
                <span>PIPELINE PROGRESS</span>
                <span>{pipelineProgress}%</span>
              </div>
              <div className="w-full bg-[#0A0D14] h-2 rounded-full overflow-hidden border border-white/5">
                <div
                  className="bg-cyan-500 h-full rounded-full transition-all duration-350"
                  style={{ width: `${pipelineProgress}%` }}
                />
              </div>

              {/* In-Memory logs */}
              <div className="bg-[#0A0D14] border border-white/5 p-3 rounded font-mono text-[9px] text-slate-405 h-[170px] overflow-y-auto space-y-2 leading-relaxed select-all">
                {pipelineLogs.map((log, index) => (
                  <p key={index} className={log.includes('SUCCESS') ? 'text-emerald-400' : log.includes('ERR') ? 'text-fuchsia-400' : 'text-slate-350'}>
                    {log}
                  </p>
                ))}
              </div>
            </div>
          )}
        </div>

        {/* Feature store record list display */}
        <div className="lg:col-span-3 bg-[#121826] border border-white/5 rounded-xl p-5 shadow-sm space-y-4">
          <div className="flex justify-between items-center flex-wrap gap-2">
            <div>
              <h3 className="font-semibold text-slate-200 text-sm">
                Active PostgreSQL Feature Store Registry Viewer
              </h3>
              <p className="text-slate-400 text-xs">
                Browse pre-computed indices. Pointwise, Pairwise, and Listwise rankers query this registry cache to sort lists instantly.
              </p>
            </div>
            {storeRecords.length > 0 && (
              <span className="bg-[#0A0D14] text-emerald-400 border border-emerald-900/40 px-2.5 py-1 text-[10px] font-mono rounded font-bold">
                ● Connected DB Store
              </span>
            )}
          </div>

          {storeRecords.length === 0 ? (
            <div className="border border-white/5 rounded-lg p-12 text-center text-slate-500">
              <p className="text-xs">Feature store is uninitialized.</p>
              <p className="text-[11px] text-slate-600 mt-1">Please launch the ETL Ingestion pipeline using the controller on the left side to register catalog features.</p>
            </div>
          ) : (
            <div className="overflow-x-auto border border-white/5 rounded-lg">
              <table className="w-full text-left border-collapse text-xs font-mono">
                <thead>
                  <tr className="bg-[#0A0D14] text-slate-405 border-b border-white/5">
                    <th className="p-3">PRODUCT ID</th>
                    <th className="p-3">CATEGORY</th>
                    <th className="p-3 text-center">BM25 INDEX</th>
                    <th className="p-3 text-center">SEMANTIC TF-IDF</th>
                    <th className="p-3 text-center">COSINE METRIC</th>
                    <th className="p-3 text-center">CTR VALUE</th>
                    <th className="p-3 text-center">ENGAGEMENT</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-white/5 text-slate-300">
                  {storeRecords.map((rec) => (
                    <tr key={rec.productId} className="hover:bg-white/5 transition-colors">
                      <td className="p-3 font-semibold text-cyan-400">
                        {rec.productId}
                      </td>
                      <td className="p-3">
                        <span className="bg-[#0A0D14] border border-white/10 text-slate-305 px-2 py-0.5 rounded text-[10px]">
                          {rec.category}
                        </span>
                      </td>
                      <td className="p-3 text-center text-slate-200">
                        {rec.features.bm25}
                      </td>
                      <td className="p-3 text-center text-slate-200">
                        {rec.features.tfidf}
                      </td>
                      <td className="p-3 text-center text-slate-200">
                        {rec.features.cosine}
                      </td>
                      <td className="p-3 text-center text-slate-200">
                        {rec.features.ctr}%
                      </td>
                      <td className="p-3 text-center text-amber-500">
                        {rec.features.engagement}⭐
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
