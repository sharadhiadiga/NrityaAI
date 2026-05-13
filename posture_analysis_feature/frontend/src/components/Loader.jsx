import { motion } from 'framer-motion'

export function Loader({ label = 'Analyzing…' }) {
  return (
    <div className="flex flex-col items-center justify-center gap-4 py-10">
      <motion.div
        className="h-14 w-14 rounded-full border-2 border-gold-500/30 border-t-gold-400"
        animate={{ rotate: 360 }}
        transition={{ repeat: Infinity, duration: 0.9, ease: 'linear' }}
      />
      <motion.p
        className="text-sm tracking-wide text-cream-muted"
        animate={{ opacity: [0.65, 1, 0.65] }}
        transition={{ repeat: Infinity, duration: 1.6, ease: 'easeInOut' }}
      >
        {label}
      </motion.p>
    </div>
  )
}
