// Debug API client for service statuses

import { fetchJson } from '../services/api'

export interface ServiceStatus {
  name: string
  label: string
  enabled: boolean
  running: boolean
  current_op: string | null
  detail: Record<string, unknown>
}

export interface DebugServicesResponse {
  services: ServiceStatus[]
}

export const getServiceStatuses = () =>
  fetchJson<DebugServicesResponse>('/debug/services')