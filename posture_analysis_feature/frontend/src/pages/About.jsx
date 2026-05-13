import { motion } from 'framer-motion'

export function About() {
  return (
    <div className="mx-auto max-w-3xl space-y-10">
      <div>
        <p className="text-xs font-semibold uppercase tracking-[0.3em] text-gold-300">About NrityaAI</p>
        <h1 className="mt-3 font-display text-4xl text-cream md:text-5xl">Tradition, translated by light</h1>
        <p className="mt-5 text-base leading-relaxed text-cream-muted">
          NrityaAI celebrates Bharatanatyam — its lines, triangles, and rhythmic stillness — while
          borrowing the clarity of modern machine perception. The interface is deliberately ornate yet
          calm: a stage for dancers, teachers, and technologists exploring what classical form looks
          like through contemporary sensors.
        </p>
      </div>

      <motion.div
        initial={{ opacity: 0, y: 12 }}
        animate={{ opacity: 1, y: 0 }}
        className="glass-panel space-y-4 p-8 text-left"
      >
        <h2 className="font-display text-2xl text-cream">How it works</h2>
        <ol className="list-decimal space-y-3 pl-5 text-sm text-cream-muted">
          <li>Capture or upload an image containing a clear full-body pose.</li>
          <li>The FastAPI service extracts landmarks and compares them to trained repertoire signals.</li>
          <li>Confidence, posture score, and textual feedback return to the React studio in real time.</li>
        </ol>
      </motion.div>

      <div className="grid gap-4 md:grid-cols-2">
        {[
          { k: 'Frontend', v: 'React, Router, Framer Motion, Tailwind, Axios' },
          { k: 'Backend', v: 'FastAPI on http://127.0.0.1:8000' },
          { k: 'Aesthetic', v: 'Maroon depth, gold filigree, glass panels' },
          { k: 'Ethos', v: 'Respectful fusion — AI assists, the artist leads.' },
        ].map((row) => (
          <motion.div
            key={row.k}
            whileHover={{ y: -2 }}
            className="rounded-2xl border border-gold-500/15 bg-black/25 p-5 text-left"
          >
            <p className="text-xs uppercase tracking-[0.2em] text-gold-400">{row.k}</p>
            <p className="mt-2 text-sm text-cream">{row.v}</p>
          </motion.div>
        ))}
      </div>
    </div>
  )
}
