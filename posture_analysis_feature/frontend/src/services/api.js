import axios from 'axios'
import { API_BASE_URL } from '../utils/constants'

export const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 45_000,
  headers: {
    Accept: 'application/json',
  },
})

api.interceptors.response.use(
  (res) => res,
  (err) => {
    const message =
      err.response?.data?.detail ??
      err.response?.data?.message ??
      err.message ??
      'Request failed'
    return Promise.reject(new Error(typeof message === 'string' ? message : JSON.stringify(message)))
  }
)
