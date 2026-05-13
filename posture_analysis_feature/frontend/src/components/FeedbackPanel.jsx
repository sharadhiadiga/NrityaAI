import { AnimatePresence, motion } from 'framer-motion'

export function FeedbackPanel({ items = [], title = 'AI feedback' }) {
  const list = Array.isArray(items) ? items : []
  return (
    <div className="glass-panel p-5 text-left">
      <p className="text-xs font-semibold uppercase tracking-[0.2em] text-gold-400/90">{title}</p>
      <ul className="mt-4 space-y-3">
        <AnimatePresence initial={false}>
          {list.length === 0 ? (
            <motion.li
              key="empty"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              className="text-sm text-cream-muted"
            >
              Align in frame — guidance will appear here.
            </motion.li>
          ) : (
            list.map((t, i) => (
              <motion.li
                key={`${i}-${t}`}
                initial={{ opacity: 0, x: -8 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: 8 }}
                transition={{ delay: i * 0.04 }}
                className="flex gap-3 rounded-xl border border-gold-500/10 bg-black/20 px-4 py-3 text-sm text-cream"
              >
                <span className="mt-0.5 h-1.5 w-1.5 shrink-0 rounded-full bg-gold-400 shadow-[0_0_12px_rgba(232,197,71,0.8)]" />
                <span>{t}</span>
              </motion.li>
            ))
          )}
        </AnimatePresence>
      </ul>
    </div>
  )
}
