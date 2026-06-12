/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import { useState } from 'react';
import {
  Search,
  Layers3,
  Compass,
  Cpu,
  Activity,
  Database,
  Brain,
  Layers,
  ChevronRight,
  Info,
  ExternalLink,
  BookOpen
} from 'lucide-react';

// Import our modular dashboards
import SearchDashboard from './components/SearchDashboard';
import RankingDashboard from './components/RankingDashboard';
import RecommendationDashboard from './components/RecommendationDashboard';
import ShapDashboard from './components/ShapDashboard';
import MlflowDashboard from './components/MlflowDashboard';
import DatasetFeatures from './components/DatasetFeatures';

export default function App() {
  const [activeTab, setActiveTab] = useState<'search' | 'ranking' | 'recommend' | 'shap' | 'mlflow' | 'dataset'>('search');

  const navigationItems = [
    { id: 'search', label: 'Search Retrieval', icon: Search, desc: 'BM25 Candidates & LTR sort' },
    { id: 'ranking', label: 'Learning To Rank', icon: Layers3, desc: 'LambdaMART / RankNet model training' },
    { id: 'recommend', label: 'Recommendation Hub', icon: Compass, desc: 'Matrix Factorization SGD & Collaborative' },
    { id: 'shap', label: 'Explainable AI', icon: Cpu, desc: 'Shapley additive values & waterfalls' },
    { id: 'mlflow', label: 'MLflow & Registry', icon: Activity, desc: 'Parameters tracing & performance comparison' },
    { id: 'dataset', label: 'MSLR Ingestion & Feature Store', icon: Database, desc: 'SVMLight preprocessing & store' },
  ];

  return (
    <div className="min-h-screen bg-[#0A0D14] text-slate-100 flex flex-col font-sans selection:bg-cyan-500/30 select-none">
      {/* Prime Header brand */}
      <header className="bg-[#0B0F19] border-b border-white/5 shrink-0 sticky top-0 z-40">
        <div className="max-w-[1600px] mx-auto px-6 py-4 flex items-center justify-between flex-wrap gap-4">
          <div className="flex items-center gap-3">
            <div className="bg-cyan-500/10 border border-cyan-500/20 p-2 rounded-lg text-cyan-400">
              <Brain className="w-6 h-6 animate-pulse" />
            </div>
            <div>
              <h1 className="text-lg font-bold text-slate-150 tracking-tight flex items-center gap-2">
                AERA-ML Search & Recommendation
              </h1>
              <span className="text-[10px] uppercase tracking-wider font-mono text-cyan-400 font-bold block">
                Enterprise Production Sandbox / Amazon ML Engineering Standard
              </span>
            </div>
          </div>

          <div className="flex items-center gap-4 text-xs font-mono text-slate-400">
            <div className="bg-[#121826] border border-white/5 px-3.5 py-1.5 rounded-lg flex items-center gap-2">
              <span className="w-2 h-2 rounded-full bg-emerald-500 shadow-[0_0_8px_rgba(16,185,129,0.6)] animate-pulse" />
              <span>Express Engine on Port 3000</span>
            </div>
          </div>
        </div>
      </header>

      {/* Main layout container with sidebar */}
      <div className="flex-1 max-w-[1600px] w-full mx-auto px-6 py-6 flex flex-col lg:flex-row gap-6">
        {/* Navigation Sidebar */}
        <aside className="lg:w-[320px] shrink-0 space-y-6">
          <div className="bg-[#0F1420] border border-white/5 rounded-xl p-4 shadow-sm">
            <p className="text-[10px] uppercase font-bold text-slate-500 font-mono tracking-widest mb-4 px-2">
              System Control Center
            </p>
            <nav className="space-y-1" id="sidebar-navigation">
              {navigationItems.map((item) => {
                const Icon = item.icon;
                const isActive = activeTab === item.id;
                return (
                  <button
                    key={item.id}
                    onClick={() => setActiveTab(item.id as any)}
                    className={`w-full text-left p-3.5 rounded-lg transition-all flex items-center justify-between gap-3 group ${
                      isActive
                        ? 'bg-cyan-500/10 text-cyan-400 border border-cyan-500/20 shadow-none font-medium'
                        : 'text-slate-400 hover:text-slate-100 hover:bg-white/5'
                    }`}
                    id={`nav-btn-${item.id}`}
                  >
                    <div className="flex items-center gap-3">
                      <Icon className={`w-5 h-5 shrink-0 ${isActive ? 'text-cyan-400' : 'text-slate-400 group-hover:text-cyan-400'}`} />
                      <div>
                        <p className="text-xs font-bold leading-none">{item.label}</p>
                        <p className={`text-[10px] mt-1 truncate max-w-[180px] ${isActive ? 'text-cyan-300' : 'text-slate-500'}`}>
                          {item.desc}
                        </p>
                      </div>
                    </div>
                    <ChevronRight className={`w-3.5 h-3.5 transition-transform ${isActive ? 'translate-x-1 text-cyan-400' : 'text-slate-500 group-hover:text-slate-200'}`} />
                  </button>
                );
              })}
            </nav>
          </div>

          {/* Quick Info Brief */}
          <div className="bg-[#121826] border border-white/5 rounded-xl p-5 shadow-sm text-xs text-slate-400 space-y-3.5">
            <h4 className="font-semibold text-slate-200 flex items-center gap-1.5 text-xs uppercase font-mono">
              <BookOpen className="w-4 h-4 text-cyan-400" />
              Architecture Brief
            </h4>
            <p className="leading-relaxed text-[11px]">
              This LTR suite leverages pointwise regressors, pairwise preference pairs, and multi-tree regression algorithms (LambdaMART) with listwise gradient updates targeting NDCG metrics optimization.
            </p>
            <div className="pt-2 border-t border-white/5 flex justify-between text-[10px] font-mono font-medium">
              <span>Seeded Database</span>
              <span className="text-emerald-400 font-bold uppercase">PostgreSQL Sim</span>
            </div>
            <div className="flex justify-between text-[10px] font-mono font-medium">
              <span>Cache Layer</span>
              <span className="text-amber-500 font-bold uppercase">Redis Sim</span>
            </div>
          </div>
        </aside>

        {/* Dynamic Display area */}
        <main className="flex-1 min-w-0" id="main-display-window">
          {activeTab === 'search' && <SearchDashboard />}
          {activeTab === 'ranking' && <RankingDashboard />}
          {activeTab === 'recommend' && <RecommendationDashboard />}
          {activeTab === 'shap' && <ShapDashboard />}
          {activeTab === 'mlflow' && <MlflowDashboard />}
          {activeTab === 'dataset' && <DatasetFeatures />}
        </main>
      </div>

      {/* Humble professional footer */}
      <footer className="bg-[#0B0F19] border-t border-white/5 text-slate-500 text-xs py-5 shrink-0 font-mono mt-12">
        <div className="max-w-[1600px] w-full mx-auto px-6 flex justify-between items-center flex-wrap gap-4">
          <p>© 2026 Amazon search and recommendation engine engineering. Reference implementation.</p>
          <div className="flex gap-4">
            <span className="text-[11px] text-slate-500 bg-[#121826] px-2 py-0.5 rounded border border-white/5">
              MSLR-WEB10K Compliant
            </span>
            <span className="text-[11px] text-slate-500 bg-[#121826] px-2 py-0.5 rounded border border-white/5">
              MLflow Model Registered
            </span>
          </div>
        </div>
      </footer>
    </div>
  );
}
