import { motion } from 'framer-motion'

export function ConfidenceBar({ value, className = '' }) {
  const pct = Math.round(Math.min(1, Math.max(0, value)) * 100)
  return (
    <div className={`space-y-2 ${className}`}>
      <div className="flex items-center justify-between text-xs text-cream-muted">
        <span>Confidence</span>
        <span className="font-semibold text-gold-300">{pct}%</span>
      </div>
      <div className="h-2.5 w-full overflow-hidden rounded-full bg-black/40 ring-1 ring-gold-500/20">
        <motion.div
          className="h-full rounded-full bg-gradient-to-r from-maroon-800 via-gold-600 to-gold-400"
          initial={false}
          animate={{ width: `${pct}%` }}
          transition={{ type: 'spring', stiffness: 120, damping: 18 }}
        />
      </div>
    </div>
  )
}
