import { api } from './api'

/**
 * @param {Blob} imageBlob
 * @returns {Promise<{ pose: string, confidence: number, matched: boolean, posture_score: number, feedback: string[] } | { error: string }>}
 */
export async function analyzePostureImage(imageBlob) {
  const form = new FormData()
  form.append('file', imageBlob, 'frame.jpg')
  const { data } = await api.post('/analyze-posture', form, {
    headers: { 'Content-Type': 'multipart/form-data' },
  })
  return data
}
