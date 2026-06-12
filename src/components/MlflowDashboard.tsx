/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import { useState, useEffect } from 'react';
import { Activity, RefreshCw, Trash2, CheckCircle, Database, GitCommit, LineChart as LucideChart } from 'lucide-react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend } from 'recharts';
import { ExperimentRun } from '../types';

export default function MlflowDashboard() {
  const [runs, setRuns] = useState<ExperimentRun[]>([]);
  const [modelRegistry, setModelRegistry] = useState<any[]>([]);
  const [loading, setLoading] = useState(false);
  const [statusMessage, setStatusMessage] = useState('');

  const fetchRuns = async () => {
    setLoading(true);
    try {
      const res = await fetch('/api/experiments');
      const data = await res.json();
      setRuns(data.runs || []);
      setModelRegistry(data.modelRegistry || []);
    } catch (err) {
      console.error('Error in fetching MLflow runs:', err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchRuns();
  }, []);

  const handleDeleteRun = async (runId: string) => {
    try {
      const res = await fetch(`/api/experiments/${runId}`, { method: 'DELETE' });
      if (res.ok) {
        setRuns(prev => prev.filter(r => r.runId !== runId));
        setStatusMessage(`Successfully pruned run ${runId} from MLflow backend registry database.`);
        setTimeout(() => setStatusMessage(''), 4000);
      }
    } catch (err) {
      console.error('Error pruning run:', err);
    }
  };

  // Convert current runs to a comparison schema for the dashboard chart
  const comparisonChartData = runs.map(run => ({
    name: run.name.replace('baseline_', '').replace('_nn', '').replace('_production_lightgbm', ''),
    'NDCG@5': run.metrics.ndcg5,
    'NDCG@10': run.metrics.ndcg10,
    MAP: run.metrics.map,
    MRR: run.metrics.mrr
  }));

  return (
    <div className="space-y-6" id="mlflow-dashboard-container">
      {/* Informational registry headers */}
      <div className="bg-[#121826] border border-white/5 rounded-xl p-6 shadow-sm">
        <div className="flex justify-between items-start gap-4 flex-wrap">
          <div>
            <h2 className="text-xl font-semibold text-slate-100 flex items-center gap-2 mb-2">
              <Activity className="w-5 h-5 text-cyan-400" />
              MLflow Experiment Tracking & Production Model Registry
            </h2>
            <p className="text-slate-400 text-sm leading-relaxed">
              Log model hyperparameters and accuracy tracking parameters automatically upon any retraining event. Compare live runs in a unified parallel metrics chart.
            </p>
          </div>
          <button
            onClick={fetchRuns}
            disabled={loading}
            className="bg-[#0A0D14] hover:bg-white/5 text-cyan-400 font-mono text-xs border border-white/10 px-3.5 py-2 rounded-lg transition-all flex items-center gap-1.5 cursor-pointer"
          >
            <RefreshCw className={`w-3.5 h-3.5 ${loading ? 'animate-spin' : ''}`} />
            Refresh MLflow Backend
          </button>
        </div>

        {statusMessage && (
          <div className="mt-4 p-3 bg-[#0A0D14] border border-white/5 text-emerald-300 text-xs rounded-lg font-mono flex items-center gap-2 animate-pulse">
            <CheckCircle className="w-4 h-4 text-emerald-400" />
            {statusMessage}
          </div>
        )}
      </div>

      {/* Grid: Graph Comparison of metrics alongside model storage registries */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Unified bar chart comparison of runs values */}
        <div className="lg:col-span-2 bg-[#121826] border border-white/5 rounded-xl p-5 shadow-sm space-y-3">
          <h3 className="font-semibold text-slate-200 text-sm flex items-center gap-2 border-b border-white/5 pb-2">
            <LucideChart className="w-4 h-4 text-cyan-400" />
            Parallel Experiment Metrics Comparison View
          </h3>
          <div className="h-[250px] w-full pt-4">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={comparisonChartData}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                <XAxis dataKey="name" stroke="#475569" fontSize={9} />
                <YAxis stroke="#475569" fontSize={9} />
                <Tooltip contentStyle={{ backgroundColor: '#0A0D14', borderColor: 'rgba(255,255,255,0.1)' }} />
                <Legend wrapperStyle={{ fontSize: 9 }} />
                <Bar dataKey="NDCG@5" fill="#22d3ee" radius={[2, 2, 0, 0]} />
                <Bar dataKey="NDCG@10" fill="#34d399" radius={[2, 2, 0, 0]} />
                <Bar dataKey="MAP" fill="#e879f9" radius={[2, 2, 0, 0]} />
                <Bar dataKey="MRR" fill="#f59e0b" radius={[2, 2, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Model registry and production staging paths */}
        <div className="lg:col-span-1 bg-[#121826] border border-white/5 rounded-xl p-5 shadow-sm space-y-4">
          <h3 className="font-semibold text-slate-200 text-sm flex items-center gap-2 border-b border-white/5 pb-2">
            <Database className="w-4 h-4 text-emerald-400" />
            Amazon-grade Model Registry
          </h3>
          <p className="text-slate-400 text-xs leading-normal">
            Models loaded here represent approved configurations that are either "Staging" or serving "Active" client traffic.
          </p>

          <div className="space-y-3 font-mono text-xs">
            {modelRegistry.map((item, index) => (
              <div key={item.modelName + index} className="bg-[#0A0D14] border border-white/5 p-3.5 rounded-lg space-y-2">
                <div className="flex justify-between items-center">
                  <span className="font-bold text-slate-200 text-xs truncate max-w-[170px]">
                    {item.modelName}
                  </span>
                  <span className={`px-2 py-0.5 rounded-full text-[9px] font-bold ${item.status === 'Active' ? 'bg-emerald-950/40 text-emerald-400 border border-emerald-900/30' : 'bg-amber-950/40 text-amber-500 border border-amber-900/30'}`}>
                    {item.status}
                  </span>
                </div>
                <div className="flex justify-between text-[11px] text-slate-500">
                  <span>VERSION: {item.version}</span>
                  <span className="text-cyan-400 font-bold">{item.accuracy}</span>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Comprehensive active experiment listing in MLflow */}
      <div className="bg-[#121826] border border-white/5 rounded-xl p-5 shadow-sm space-y-4">
        <h3 className="font-semibold text-slate-200 text-sm">
          Active MLflow Runs database
        </h3>

        <div className="overflow-x-auto border border-white/5 rounded-lg text-xs font-mono">
          <table className="w-full text-left border-collapse">
            <thead>
              <tr className="bg-[#0A0D14] text-slate-500 border-b border-white/5">
                <th className="p-3.5 font-medium">RUN_ID</th>
                <th className="p-3.5 font-medium">EXPERIMENT RUN NAME</th>
                <th className="p-3.5 font-medium">TIMESTAMP</th>
                <th className="p-3.5 font-medium">PARADIGM</th>
                <th className="p-3.5 font-medium">HYPERPARAMETERS LOGGED</th>
                <th className="p-3.5 font-medium text-center">NDCG@5</th>
                <th className="p-3.5 font-medium text-center">ACTION</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-white/5 text-slate-300">
              {runs.map((run) => (
                <tr key={run.runId} className="hover:bg-white/5 transition-colors">
                  <td className="p-3.5 text-cyan-400 font-bold">
                    {run.runId}
                  </td>
                  <td className="p-3.5 font-sans font-medium text-slate-100 flex items-center gap-1.5">
                    <GitCommit className="w-4 h-4 text-slate-500" />
                    {run.name}
                  </td>
                  <td className="p-3.5 text-slate-500">
                    {new Date(run.timestamp).toLocaleString()}
                  </td>
                  <td className="p-3.5 uppercase text-slate-400 text-[11px]">
                    {run.algorithm.replace('_', ' ')}
                  </td>
                  <td className="p-3.5 text-slate-400 truncate max-w-[200px]" title={JSON.stringify(run.parameters)}>
                    {JSON.stringify(run.parameters)}
                  </td>
                  <td className="p-3.5 text-center text-emerald-400 font-bold">
                    {run.metrics.ndcg5}
                  </td>
                  <td className="p-3.5 text-center">
                    <button
                      onClick={() => handleDeleteRun(run.runId)}
                      className="text-rose-500 hover:text-rose-455 p-1 hover:bg-rose-950/20 rounded transition-colors cursor-pointer"
                      title="Prune experiment"
                    >
                      <Trash2 className="w-4 h-4" />
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
