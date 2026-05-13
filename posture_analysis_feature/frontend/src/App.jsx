import { BrowserRouter, Navigate, Route, Routes } from 'react-router-dom'
import { AppLayout } from './components/AppLayout'
import { DashboardProvider } from './hooks/useDashboard'
import { About } from './pages/About'
import { Dashboard } from './pages/Dashboard'
import { Home } from './pages/Home'
import { Training } from './pages/Training'

export default function App() {
  return (
    <BrowserRouter>
      <DashboardProvider>
        <Routes>
          <Route element={<AppLayout />}>
            <Route index element={<Home />} />
            <Route path="training" element={<Training />} />
            <Route path="dashboard" element={<Dashboard />} />
            <Route path="about" element={<About />} />
            <Route path="*" element={<Navigate to="/" replace />} />
          </Route>
        </Routes>
      </DashboardProvider>
    </BrowserRouter>
  )
}
