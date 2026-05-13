import { useCallback, useEffect, useRef, useState } from 'react'
import { motion } from 'framer-motion'
import { CameraFeed } from '../components/CameraFeed'
import { ConfidenceBar } from '../components/ConfidenceBar'
import { FeedbackPanel } from '../components/FeedbackPanel'
import { Loader } from '../components/Loader'
import { PoseCard } from '../components/PoseCard'
import { useDashboard } from '../hooks/useDashboard'
import { useLivePosePrediction } from '../hooks/useLivePosePrediction'
import { usePoseAnalysis } from '../hooks/usePoseAnalysis'
import { JPEG_QUALITY } from '../utils/constants'

const statusCopy = {
  idle: 'Ready',
  paused: 'Paused',
  capturing: 'Capturing frame',
  analyzing: 'Model inferencing',
  live: 'Live',
  no_frame: 'No camera frame',
  api_error: 'API message',
  network_error: 'Network issue',
}

export function Training() {
  const videoRef = useRef(null)
  const canvasRef = useRef(null)
  const fileRef = useRef(null)

  const { recordSnapshot } = useDashboard()
  const { analyze, loading: manualLoading, error: manualError, resetError } = usePoseAnalysis()

  const [live, setLive] = useState(false)
  const [camReady, setCamReady] = useState(false)
  const [manual, setManual] = useState(null)

  const onSnapshot = useCallback(
    (payload) => {
      recordSnapshot(payload)
    },
    [recordSnapshot]
  )

  const getFrameBlob = useCallback(async () => {
    const video = videoRef.current
    const canvas = canvasRef.current
    if (!video || !canvas || video.readyState < 2) return null
    const w = video.videoWidth
    const h = video.videoHeight
    if (!w || !h) return null
    canvas.width = w
    canvas.height = h
    const ctx = canvas.getContext('2d')
    if (!ctx) return null
    ctx.drawImage(video, 0, 0, w, h)
    return await new Promise((resolve) => {
      canvas.toBlob((b) => resolve(b), 'image/jpeg', JPEG_QUALITY)
    })
  }, [])

  const { display, status, resetSession } = useLivePosePrediction(
    live && camReady,
    getFrameBlob,
    onSnapshot
  )

  useEffect(() => {
    let stream
    ;(async () => {
      try {
        stream = await navigator.mediaDevices.getUserMedia({
          video: { facingMode: 'user', width: { ideal: 1280 }, height: { ideal: 720 } },
          audio: false,
        })
        const v = videoRef.current
        if (v) {
          v.srcObject = stream
          await v.play().catch(() => {})
        }
        setCamReady(true)
      } catch {
        setCamReady(false)
      }
    })()
    return () => {
      stream?.getTracks().forEach((t) => t.stop())
      setCamReady(false)
    }
  }, [])

  const handleUpload = async (e) => {
    const file = e.target.files?.[0]
    e.target.value = ''
    if (!file) return
    resetError()
    setManual(null)
    try {
      const data = await analyze(file)
      setManual(data)
      recordSnapshot({
        pose: data.pose,
        confidence: data.confidence,
        posture_score: data.posture_score,
        feedback: data.feedback,
        matched: data.matched,
        rawPose: data.pose,
      })
    } catch {
      /* surfaced via manualError */
    }
  }

  const runManualOnFrame = async () => {
    resetError()
    setManual(null)
    const blob = await getFrameBlob()
    if (!blob) return
    try {
      const data = await analyze(blob)
      setManual(data)
      recordSnapshot({
        pose: data.pose,
        confidence: data.confidence,
        posture_score: data.posture_score,
        feedback: data.feedback,
        matched: data.matched,
        rawPose: data.pose,
      })
    } catch {
      /* manualError */
    }
  }

  const livePose = live ? display.pose : manual?.pose ?? '—'
  const liveConf = live ? display.confidence : manual?.confidence ?? 0
  const liveScore = live ? display.posture_score : manual?.posture_score ?? null
  const liveFb = live ? display.feedback : manual?.feedback ?? []
  const liveMatched = live ? display.matched : manual?.matched ?? false
  const rawHint = live ? `Instant: ${display.rawPose}` : manual ? `Instant: ${manual.pose}` : null

  return (
    <div className="space-y-8">
      <div>
        <p className="text-xs font-semibold uppercase tracking-[0.3em] text-gold-300">Training studio</p>
        <h1 className="mt-2 font-display text-4xl text-cream md:text-5xl">Live posture atelier</h1>
        <p className="mt-3 max-w-2xl text-sm text-cream-muted">
          Webcam frames post to <code className="text-gold-200">POST /analyze-posture</code> as{' '}
          <code className="text-gold-200">multipart/form-data</code> every 800ms while live mode is on.
        </p>
      </div>

      <div className="grid gap-8 lg:grid-cols-2">
        <div className="space-y-4">
          <CameraFeed
            ref={videoRef}
            mirrored
            status={live ? status : 'idle'}
            badge={live ? statusCopy[status] ?? status : 'Ready'}
            footer={
              <div className="flex flex-wrap items-center gap-3">
                <motion.button
                  type="button"
                  whileTap={{ scale: 0.97 }}
                  onClick={() => {
                    if (live) resetSession()
                    setLive((v) => !v)
                  }}
                  className={`rounded-full px-5 py-2 text-xs font-semibold uppercase tracking-widest ring-1 transition-colors ${
                    live
                      ? 'bg-emerald-500/20 text-emerald-100 ring-emerald-400/40'
                      : 'bg-white/10 text-cream ring-gold-500/30'
                  }`}
                >
                  {live ? 'Stop live' : 'Start live'}
                </motion.button>
                <motion.button
                  type="button"
                  whileTap={{ scale: 0.97 }}
                  onClick={runManualOnFrame}
                  disabled={manualLoading || !camReady}
                  className="rounded-full bg-gradient-to-r from-gold-500 to-gold-400 px-5 py-2 text-xs font-semibold uppercase tracking-widest text-maroon-950 shadow-md disabled:opacity-40"
                >
                  Analyze frame
                </motion.button>
                <input ref={fileRef} type="file" accept="image/*" className="hidden" onChange={handleUpload} />
                <motion.button
                  type="button"
                  whileTap={{ scale: 0.97 }}
                  onClick={() => fileRef.current?.click()}
                  className="rounded-full border border-gold-500/35 bg-black/30 px-5 py-2 text-xs font-semibold uppercase tracking-widest text-cream"
                >
                  Upload image
                </motion.button>
              </div>
            }
          />
          <canvas ref={canvasRef} className="hidden" />
          {!camReady ? (
            <p className="text-center text-sm text-amber-200/90">
              Camera permission is required for the live studio. You can still upload images.
            </p>
          ) : null}
          {manualError ? (
            <p className="text-center text-sm text-rose-300">{manualError}</p>
          ) : null}
        </div>

        <div className="space-y-4">
          <PoseCard
            pose={String(livePose)}
            matched={Boolean(liveMatched)}
            subtitle={rawHint}
            title="Real-time pose"
          />
          <div className="glass-panel space-y-5 p-6">
            <div className="flex items-center justify-between gap-3">
              <p className="text-xs font-semibold uppercase tracking-[0.2em] text-gold-400/90">
                Posture score
              </p>
              <motion.span
                key={liveScore}
                initial={{ opacity: 0, y: 6 }}
                animate={{ opacity: 1, y: 0 }}
                className="font-display text-5xl text-gradient-gold"
              >
                {liveScore ?? '—'}
              </motion.span>
            </div>
            <ConfidenceBar value={Number(liveConf) || 0} />
            {manualLoading || (live && (status === 'analyzing' || status === 'capturing')) ? (
              <Loader label="Invoking NrityaAI…" />
            ) : null}
          </div>
          <FeedbackPanel items={liveFb} />
        </div>
      </div>
    </div>
  )
}
