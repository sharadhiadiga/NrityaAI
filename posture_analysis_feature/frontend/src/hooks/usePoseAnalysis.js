import { useCallback, useState } from 'react'
import { analyzePostureImage } from '../services/postureApi'

export function usePoseAnalysis() {
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  const analyze = useCallback(async (blob) => {
    setLoading(true)
    setError(null)
    try {
      const data = await analyzePostureImage(blob)
      if (data?.error) throw new Error(data.error)
      return data
    } catch (e) {
      const msg = e instanceof Error ? e.message : 'Analysis failed'
      setError(msg)
      throw e
    } finally {
      setLoading(false)
    }
  }, [])

  const resetError = useCallback(() => setError(null), [])

  return { analyze, loading, error, resetError }
}
