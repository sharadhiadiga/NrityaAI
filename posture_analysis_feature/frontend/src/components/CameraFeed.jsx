import { forwardRef } from 'react'
import { motion } from 'framer-motion'

export const CameraFeed = forwardRef(function CameraFeed(
  { mirrored = true, status = 'idle', badge, footer, className = '' },
  ref
) {
  const tone =
    status === 'live'
      ? 'bg-emerald-400/90'
      : status === 'analyzing' || status === 'capturing'
        ? 'bg-amber-400/90'
        : status === 'network_error' || status === 'api_error'
          ? 'bg-rose-400/90'
          : 'bg-white/40'

  return (
    <motion.div
      layout
      className={`relative overflow-hidden rounded-2xl ring-1 ring-gold-500/25 ${className}`}
      style={{
        background:
          'linear-gradient(145deg, rgba(255,255,255,0.06), rgba(0,0,0,0.35))',
      }}
      whileHover={{ scale: 1.005 }}
      transition={{ type: 'spring', stiffness: 280, damping: 24 }}
    >
      <div className="pointer-events-none absolute inset-0 bg-gradient-to-tr from-gold-500/10 via-transparent to-maroon-800/30" />
      <video
        ref={ref}
        playsInline
        muted
        autoPlay
        className={`aspect-video w-full object-cover ${mirrored ? 'scale-x-[-1]' : ''}`}
      />
      <div className="pointer-events-none absolute inset-x-0 top-0 flex items-start justify-between p-4">
        <div className="flex items-center gap-2 rounded-full bg-black/45 px-3 py-1.5 text-xs text-cream backdrop-blur-md ring-1 ring-white/10">
          <span className={`h-2 w-2 rounded-full ${tone} shadow-[0_0_12px_currentColor]`} />
          <span className="tracking-wide">{badge ?? 'Camera'}</span>
        </div>
      </div>
      {footer ? (
        <div className="absolute inset-x-0 bottom-0 bg-gradient-to-t from-black/70 to-transparent p-4 pt-12">
          {footer}
        </div>
      ) : null}
    </motion.div>
  )
})
