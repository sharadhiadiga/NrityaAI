import { useMemo } from 'react'
import { Link } from 'react-router-dom'
import { motion } from 'framer-motion'
import { useDashboard } from '../hooks/useDashboard'
import { ConfidenceBar } from '../components/ConfidenceBar'
import { FeedbackPanel } from '../components/FeedbackPanel'
import { PoseCard } from '../components/PoseCard'

export function Dashboard() {
  const { latest, history, clearHistory } = useDashboard()

  const feedbackSummary = useMemo(() => {
    const set = new Set()
    for (const h of history) {
      for (const f of h.feedback ?? []) set.add(f)
    }
    return [...set].slice(0, 8)
  }, [history])

  return (
    <div className="space-y-10">
      <div className="flex flex-col justify-between gap-4 md:flex-row md:items-end">
        <div>
          <p className="text-xs font-semibold uppercase tracking-[0.3em] text-gold-300">Dashboard</p>
          <h1 className="mt-2 font-display text-4xl text-cream md:text-5xl">Session intelligence</h1>
          <p className="mt-3 max-w-xl text-sm text-cream-muted">
            Latest detections from the training studio stream into this view — confidence, score, and
            evolving AI commentary.
          </p>
        </div>
        <motion.button
          type="button"
          className="self-start rounded-full border border-gold-500/30 bg-white/5 px-5 py-2 text-xs font-semibold uppercase tracking-widest text-cream-muted hover:border-gold-500/50 hover:text-cream"
          whileTap={{ scale: 0.97 }}
          onClick={clearHistory}
        >
          Clear history
        </motion.button>
      </div>

      {!latest ? (
        <div className="glass-panel py-16 text-center text-cream-muted">
          No session data yet. Open{' '}
          <Link className="text-gold-300 underline decoration-gold-500/40" to="/training">
            Training
          </Link>{' '}
          and run live or manual analysis to populate this board.
        </div>
      ) : (
        <div className="grid gap-6 lg:grid-cols-2">
          <PoseCard
            pose={latest.pose ?? '—'}
            matched={Boolean(latest.matched)}
            subtitle={`Model glimpse: ${latest.rawPose ?? '—'}`}
          />
          <div className="glass-panel space-y-6 p-6 text-left">
            <div>
              <p className="text-xs font-semibold uppercase tracking-[0.2em] text-gold-400/90">
                Posture score
              </p>
              <motion.p
                key={latest.posture_score}
                initial={{ scale: 0.9, opacity: 0 }}
                animate={{ scale: 1, opacity: 1 }}
                className="mt-2 font-display text-6xl text-gradient-gold"
              >
                {Math.round(Number(latest.posture_score) || 0)}
              </motion.p>
            </div>
            <ConfidenceBar value={Number(latest.confidence) || 0} />
          </div>
        </div>
      )}

      <div className="grid gap-6 lg:grid-cols-2">
        <FeedbackPanel items={feedbackSummary} title="AI feedback summary" />
        <div className="glass-panel p-5 text-left">
          <p className="text-xs font-semibold uppercase tracking-[0.2em] text-gold-400/90">
            Pose history
          </p>
          <ul className="mt-4 max-h-80 space-y-2 overflow-y-auto pr-1 text-sm">
            {history.length === 0 ? (
              <li className="text-cream-muted">History appears as you analyze poses.</li>
            ) : (
              history.map((h) => (
                <li
                  key={h.at}
                  className="flex items-center justify-between gap-3 rounded-xl border border-white/5 bg-black/25 px-3 py-2 text-cream"
                >
                  <span className="font-medium capitalize">{h.pose}</span>
                  <span className="shrink-0 text-xs text-cream-muted">
                    {Math.round((h.confidence ?? 0) * 100)}% · {Math.round(Number(h.posture_score) || 0)} pts
                  </span>
                </li>
              ))
            )}
          </ul>
        </div>
      </div>
    </div>
  )
}
