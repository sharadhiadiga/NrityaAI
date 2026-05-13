import { useState } from 'react'
import { Link, NavLink } from 'react-router-dom'
import { AnimatePresence, motion } from 'framer-motion'

const links = [
  { to: '/', label: 'Home' },
  { to: '/training', label: 'Training' },
  { to: '/dashboard', label: 'Dashboard' },
  { to: '/about', label: 'About' },
]

export function Navbar() {
  const [open, setOpen] = useState(false)

  return (
    <header className="sticky top-0 z-50 border-b border-gold-500/10 bg-maroon-950/70 backdrop-blur-xl">
      <div className="mx-auto flex max-w-6xl items-center justify-between gap-4 px-4 py-4 md:px-6">
        <Link to="/" className="group flex items-center gap-3">
          <motion.div
            className="grid h-11 w-11 place-items-center rounded-2xl bg-gradient-to-br from-gold-400 to-gold-600 text-maroon-950 shadow-lg shadow-gold-500/20"
            whileHover={{ rotate: -6, scale: 1.05 }}
            transition={{ type: 'spring', stiffness: 400, damping: 18 }}
          >
            <span className="font-display text-xl font-bold">नृ</span>
          </motion.div>
          <div className="text-left leading-tight">
            <p className="font-display text-xl font-semibold text-cream">NrityaAI</p>
            <p className="text-[11px] uppercase tracking-[0.28em] text-gold-400/90">
              Posture intelligence
            </p>
          </div>
        </Link>

        <nav className="hidden items-center gap-1 md:flex">
          {links.map((l) => (
            <NavLink
              key={l.to}
              to={l.to}
              className={({ isActive }) =>
                `rounded-full px-4 py-2 text-sm font-medium transition-colors ${
                  isActive
                    ? 'bg-gold-500/15 text-gold-200 ring-1 ring-gold-500/35'
                    : 'text-cream-muted hover:bg-white/5 hover:text-cream'
                }`
              }
            >
              {l.label}
            </NavLink>
          ))}
        </nav>

        <div className="flex items-center gap-2 md:hidden">
          <motion.button
            type="button"
            aria-label="Open menu"
            className="rounded-xl border border-gold-500/25 bg-white/5 px-3 py-2 text-sm text-cream"
            whileTap={{ scale: 0.96 }}
            onClick={() => {
              setOpen((v) => !v)
            }}
          >
            Menu
          </motion.button>
        </div>
      </div>

      <AnimatePresence>
        {open ? (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            className="border-t border-gold-500/10 bg-maroon-950/95 md:hidden"
          >
            <div className="flex flex-col gap-1 px-4 py-3">
              {links.map((l) => (
                <NavLink
                  key={l.to}
                  to={l.to}
                  onClick={() => setOpen(false)}
                  className={({ isActive }) =>
                    `rounded-xl px-3 py-3 text-sm ${
                      isActive ? 'bg-gold-500/15 text-gold-100' : 'text-cream-muted'
                    }`
                  }
                >
                  {l.label}
                </NavLink>
              ))}
            </div>
          </motion.div>
        ) : null}
      </AnimatePresence>
    </header>
  )
}
