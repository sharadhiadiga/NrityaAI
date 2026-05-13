/**
 * Exponential moving average. Returns `next` when `prev` is null/undefined.
 */
export function ema(prev, next, alpha) {
  if (next == null || Number.isNaN(next)) return prev ?? 0
  if (prev == null || Number.isNaN(prev)) return next
  return alpha * next + (1 - alpha) * prev
}

export function clamp(n, min, max) {
  return Math.min(max, Math.max(min, n))
}
