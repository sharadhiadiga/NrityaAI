import { NavLink } from 'react-router-dom'
import { motion } from 'framer-motion'

const links = [
  { to: '/', label: 'Home', hint: 'Overview' },
  { to: '/training', label: 'Training', hint: 'Live studio' },
  { to: '/dashboard', label: 'Dashboard', hint: 'Insights' },
  { to: '/about', label: 'About', hint: 'Story' },
]

export function Sidebar({ className = '' }) {
  return (
    <aside
      className={`hidden lg:flex lg:w-64 lg:flex-col lg:border-r lg:border-gold-500/10 lg:bg-black/20 lg:backdrop-blur-md ${className}`}
    >
      <div className="p-6">
        <p className="text-[11px] font-semibold uppercase tracking-[0.3em] text-gold-400/80">
          Navigate
        </p>
        <nav className="mt-6 space-y-2">
          {links.map((l, i) => (
            <motion.div
              key={l.to}
              initial={{ opacity: 0, x: -10 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: 0.04 * i }}
            >
              <NavLink
                to={l.to}
                className={({ isActive }) =>
                  `group flex flex-col rounded-2xl border px-4 py-3 transition-colors ${
                    isActive
                      ? 'border-gold-500/40 bg-gold-500/10 text-cream'
                      : 'border-transparent bg-white/[0.03] text-cream-muted hover:border-gold-500/20 hover:bg-white/[0.05] hover:text-cream'
                  }`
              }
              >
                <span className="text-sm font-semibold">{l.label}</span>
                <span className="text-xs text-cream-muted group-hover:text-cream/80">{l.hint}</span>
              </NavLink>
            </motion.div>
          ))}
        </nav>
        <div className="mt-10 rounded-2xl border border-gold-500/15 bg-gradient-to-br from-maroon-900/80 to-black/40 p-4 text-xs text-cream-muted">
          <p className="font-semibold text-gold-300">Demo tip</p>
          <p className="mt-2 leading-relaxed">
            Start the FastAPI server, allow camera access, and enable live mode for continuous
            classical posture guidance.
          </p>
        </div>
      </div>
    </aside>
  )
}
