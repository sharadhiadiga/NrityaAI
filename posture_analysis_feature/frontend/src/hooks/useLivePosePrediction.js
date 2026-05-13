import { useEffect, useRef, useState } from 'react'
import {
  ANALYZE_INTERVAL_MS,
  CONFIDENCE_SWITCH_THRESHOLD,
  EMA_ALPHA,
  POSE_SWITCH_STREAK,
} from '../utils/constants'
import { ema } from '../utils/smoothing'
import { analyzePostureImage } from '../services/postureApi'

const initialDisplay = {
  pose: '—',
  confidence: 0,
  posture_score: null,
  feedback: [],
  matched: false,
  rawPose: '—',
}

/**
 * @param {boolean} enabled
 * @param {() => Promise<Blob | null>} getFrameBlob
 * @param {(payload: object) => void} [onSnapshot]
 */
export function useLivePosePrediction(enabled, getFrameBlob, onSnapshot) {
  const [display, setDisplay] = useState(initialDisplay)
  const [engineStatus, setEngineStatus] = useState('idle')

  const smoothConf = useRef(null)
  const smoothScore = useRef(null)
  const streak = useRef({ pose: null, count: 0 })
  const lockedPose = useRef('—')
  const gen = useRef(0)
  const busy = useRef(false)

  const onSnapshotRef = useRef(onSnapshot)
  const getFrameRef = useRef(getFrameBlob)

  useEffect(() => {
    onSnapshotRef.current = onSnapshot
  }, [onSnapshot])

  useEffect(() => {
    getFrameRef.current = getFrameBlob
  }, [getFrameBlob])

  useEffect(() => {
    if (!enabled) return

    const run = () => {
      const my = ++gen.current
      if (busy.current) return
      busy.current = true

      ;(async () => {
        setEngineStatus('capturing')
        try {
          const blob = await getFrameRef.current()
          if (my !== gen.current) return
          if (!blob) {
            setEngineStatus('no_frame')
            return
          }

          setEngineStatus('analyzing')
          const raw = await analyzePostureImage(blob)
          if (my !== gen.current) return

          if (raw?.error) {
            setEngineStatus('api_error')
            setDisplay((d) => ({
              ...d,
              feedback: [String(raw.error)],
              matched: false,
            }))
            return
          }

          const conf = Number(raw.confidence) || 0
          const score = Number(raw.posture_score)
          const rawPose = String(raw.pose ?? '—')

          smoothConf.current = ema(smoothConf.current, conf, EMA_ALPHA)
          smoothScore.current = ema(
            smoothScore.current,
            Number.isFinite(score) ? score : smoothScore.current ?? 0,
            EMA_ALPHA
          )

          const high = conf > CONFIDENCE_SWITCH_THRESHOLD && rawPose !== 'none'
          let nextLocked = lockedPose.current

          if (high) {
            if (streak.current.pose === rawPose) {
              streak.current = { pose: rawPose, count: streak.current.count + 1 }
            } else {
              streak.current = { pose: rawPose, count: 1 }
            }
            if (streak.current.count >= POSE_SWITCH_STREAK) {
              lockedPose.current = rawPose
              nextLocked = rawPose
            }
          } else {
            streak.current = { pose: null, count: 0 }
          }

          const next = {
            pose: nextLocked,
            confidence: smoothConf.current ?? conf,
            posture_score: Math.round(smoothScore.current ?? score ?? 0),
            feedback: Array.isArray(raw.feedback) ? raw.feedback : [],
            matched: Boolean(raw.matched),
            rawPose,
          }

          setDisplay(next)
          setEngineStatus('live')
          onSnapshotRef.current?.({
            ...next,
            confidence: smoothConf.current ?? conf,
            posture_score: smoothScore.current ?? score,
          })
        } catch {
          if (my === gen.current) setEngineStatus('network_error')
        } finally {
          busy.current = false
        }
      })()
    }

    run()
    const id = setInterval(run, ANALYZE_INTERVAL_MS)

    return () => {
      clearInterval(id)
      gen.current += 1
    }
  }, [enabled])

  const resetSession = () => {
    gen.current += 1
    smoothConf.current = null
    smoothScore.current = null
    streak.current = { pose: null, count: 0 }
    lockedPose.current = '—'
    busy.current = false
    setDisplay(initialDisplay)
    setEngineStatus('idle')
  }

  return { display, status: engineStatus, resetSession }
}
