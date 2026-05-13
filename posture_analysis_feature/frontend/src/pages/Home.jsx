import { Link } from 'react-router-dom'
import { motion } from 'framer-motion'

const container = {
  hidden: { opacity: 0 },
  show: {
    opacity: 1,
    transition: { staggerChildren: 0.08, delayChildren: 0.05 },
  },
}

const item = {
  hidden: { opacity: 0, y: 16 },
  show: { opacity: 1, y: 0, transition: { duration: 0.45, ease: [0.22, 1, 0.36, 1] } },
}

export function Home() {
  return (
    <div className="space-y-16">
      <section className="relative overflow-hidden rounded-3xl border border-gold-500/20 bg-gradient-to-br from-maroon-900/90 via-maroon-950 to-black/80 px-6 py-16 md:px-14 md:py-20">
        <div className="pointer-events-none absolute -right-24 -top-24 h-72 w-72 rounded-full bg-gold-500/10 blur-3xl" />
        <div className="pointer-events-none absolute -left-20 bottom-0 h-64 w-64 rounded-full bg-maroon-800/40 blur-3xl" />
        <motion.div
          className="pointer-events-none absolute right-10 top-10 opacity-30"
          animate={{ rotate: 360 }}
          transition={{ repeat: Infinity, duration: 120, ease: 'linear' }}
        >
          <svg width="160" height="160" viewBox="0 0 160 160" fill="none" aria-hidden>
            <circle cx="80" cy="80" r="78" stroke="url(#g)" strokeWidth="0.6" strokeDasharray="4 6" />
            <defs>
              <linearGradient id="g" x1="0" y1="0" x2="1" y2="1">
                <stop stopColor="#f6d873" />
                <stop offset="1" stopColor="#6b1f32" />
              </linearGradient>
            </defs>
          </svg>
        </motion.div>

        <div className="relative mx-auto max-w-3xl text-center">
          <motion.p
            initial={{ opacity: 0, y: 8 }}
            animate={{ opacity: 1, y: 0 }}
            className="text-xs font-semibold uppercase tracking-[0.35em] text-gold-300"
          >
            NrityaAI platform
          </motion.p>
          <motion.h1
            initial={{ opacity: 0, y: 12 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.05 }}
            className="mt-5 font-display text-5xl font-semibold leading-[1.05] text-cream md:text-6xl"
          >
            Classical line meets{' '}
            <span className="text-gradient-gold">intelligent motion</span>
          </motion.h1>
          <motion.p
            initial={{ opacity: 0, y: 12 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.12 }}
            className="mt-6 text-base leading-relaxed text-cream-muted md:text-lg"
          >
            A responsive studio for Bharatanatyam posture analysis — live webcam guidance, refined
            scoring, and AI feedback tuned for the geometry of adavus and stances.
          </motion.p>
          <motion.div
            initial={{ opacity: 0, y: 12 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.18 }}
            className="mt-10 flex flex-wrap items-center justify-center gap-4"
          >
            <Link to="/training">
              <motion.span
                className="inline-flex rounded-full bg-gradient-to-r from-gold-500 to-gold-400 px-8 py-3 text-sm font-semibold text-maroon-950 shadow-lg shadow-gold-500/25"
                whileHover={{ scale: 1.04, y: -1 }}
                whileTap={{ scale: 0.98 }}
              >
                Enter training studio
              </motion.span>
            </Link>
            <Link to="/dashboard">
              <motion.span
                className="inline-flex rounded-full border border-gold-500/40 bg-white/5 px-8 py-3 text-sm font-semibold text-cream backdrop-blur-md"
                whileHover={{ scale: 1.03, borderColor: 'rgba(212,175,55,0.65)' }}
                whileTap={{ scale: 0.98 }}
              >
                View dashboard
              </motion.span>
            </Link>
          </motion.div>
        </div>
      </section>

      <motion.section
        variants={container}
        initial="hidden"
        whileInView="show"
        viewport={{ once: true, margin: '-80px' }}
        className="grid gap-6 md:grid-cols-3"
      >
        {[
          {
            title: 'Live adavu sensing',
            body: 'Frames captured every 800ms with confidence smoothing and stable pose locking for a fluid stage feel.',
          },
          {
            title: 'Heritage palette',
            body: 'Deep maroon fields, molten gold accents, and glass panels evoke temple silk and lamp glow.',
          },
          {
            title: 'FastAPI backbone',
            body: 'Multipart analysis to /analyze-posture with graceful error surfaces and resilient loading states.',
          },
        ].map((c) => (
          <motion.div
            key={c.title}
            variants={item}
            className="glass-panel p-6 text-left transition-transform hover:-translate-y-1"
          >
            <div className="h-1 w-12 rounded-full bg-gradient-to-r from-gold-600 to-gold-300" />
            <h3 className="mt-5 font-display text-2xl text-cream">{c.title}</h3>
            <p className="mt-3 text-sm leading-relaxed text-cream-muted">{c.body}</p>
          </motion.div>
        ))}
      </motion.section>
    </div>
  )
}
