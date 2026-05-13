import { motion } from 'framer-motion'

export function PoseCard({ title = 'Detected posture', pose, matched, subtitle, className = '' }) {
  return (
    <motion.div
      layout
      className={`glass-panel p-6 text-left ${className}`}
      whileHover={{ y: -2, boxShadow: '0 16px 48px rgba(0,0,0,0.45)' }}
      transition={{ type: 'spring', stiffness: 260, damping: 22 }}
    >
      <p className="text-xs font-semibold uppercase tracking-[0.2em] text-gold-400/90">{title}</p>
      <motion.h3
        key={pose}
        initial={{ opacity: 0, y: 6 }}
        animate={{ opacity: 1, y: 0 }}
        className="mt-3 font-display text-4xl font-semibold text-cream md:text-5xl"
      >
        {pose}
      </motion.h3>
      <div className="mt-3 flex flex-wrap items-center gap-2">
        <span
          className={`rounded-full px-3 py-1 text-xs font-medium ring-1 ${
            matched
              ? 'bg-gold-500/15 text-gold-200 ring-gold-500/40'
              : 'bg-white/5 text-cream-muted ring-white/10'
          }`}
        >
          {matched ? 'Matched repertoire' : 'Exploratory / low match'}
        </span>
        {subtitle ? (
          <span className="text-xs text-cream-muted">{subtitle}</span>
        ) : null}
      </div>
    </motion.div>
  )
}
