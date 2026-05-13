export const API_BASE_URL =
  import.meta.env.VITE_API_URL ?? 'http://127.0.0.1:8000'

export const ANALYZE_INTERVAL_MS = 800
export const CONFIDENCE_SWITCH_THRESHOLD = 0.6
/** Consecutive high-confidence frames required before updating displayed pose */
export const POSE_SWITCH_STREAK = 2
export const EMA_ALPHA = 0.38
export const JPEG_QUALITY = 0.72
export const HISTORY_LIMIT = 48
