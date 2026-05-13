/* eslint-disable react-refresh/only-export-components -- provider + hook */
import { createContext, useCallback, useContext, useMemo, useState } from 'react'
import { HISTORY_LIMIT } from '../utils/constants'

const DashboardContext = createContext(null)

export function DashboardProvider({ children }) {
  const [latest, setLatest] = useState(null)
  const [history, setHistory] = useState([])

  const recordSnapshot = useCallback((snapshot) => {
    setLatest(snapshot)
    setHistory((prev) =>
      [{ ...snapshot, at: Date.now() }, ...prev].slice(0, HISTORY_LIMIT)
    )
  }, [])

  const clearHistory = useCallback(() => {
    setHistory([])
    setLatest(null)
  }, [])

  const value = useMemo(
    () => ({ latest, history, recordSnapshot, clearHistory }),
    [latest, history, recordSnapshot, clearHistory]
  )

  return (
    <DashboardContext.Provider value={value}>{children}</DashboardContext.Provider>
  )
}

export function useDashboard() {
  const ctx = useContext(DashboardContext)
  if (!ctx) throw new Error('useDashboard must be used within DashboardProvider')
  return ctx
}
