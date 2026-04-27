import { useState, useEffect } from 'react'
import * as api from '../services/api'
import { Loader2 } from 'lucide-react'
import clsx from 'clsx'

export interface ProviderModelValue {
  provider: string
  model: string
}

export interface ProviderModelSelectorProps {
  value: ProviderModelValue
  onChange: (value: ProviderModelValue) => void
  label?: string
  disabled?: boolean
  configuredOnly?: boolean
}

export default function ProviderModelSelector({
  value,
  onChange,
  label,
  disabled = false,
  configuredOnly = false,
}: ProviderModelSelectorProps) {
  const [providers, setProviders] = useState<{ name: string; display_name: string }[]>([])
  const [models, setModels] = useState<api.LLMModel[]>([])
  const [loadingProviders, setLoadingProviders] = useState(true)
  const [loadingModels, setLoadingModels] = useState(false)
  const [modelsError, setModelsError] = useState<string | null>(null)

  // Fetch providers on mount
  useEffect(() => {
    const loadProviders = async () => {
      setLoadingProviders(true)
      try {
        const provs = configuredOnly
          ? await api.getLLMProvidersFromDB()
          : await api.getLLMProviders()
        setProviders(provs || [])
      } catch (e) {
        console.error('Failed to load providers', e)
        setProviders([])
      } finally {
        setLoadingProviders(false)
      }
    }
    loadProviders()
  }, [configuredOnly])

  // Fetch models when provider changes
  useEffect(() => {
    const loadModels = async () => {
      if (!value.provider) {
        setModels([])
        return
      }
      setLoadingModels(true)
      setModelsError(null)
      try {
        const result = await api.getLLMProviderModels(value.provider)
        setModels(result.models || [])
      } catch (e) {
        console.error('Failed to load models for', value.provider, e)
        setModels([])
        setModelsError('Modelle konnten nicht geladen werden')
      } finally {
        setLoadingModels(false)
      }
    }
    loadModels()
  }, [value.provider])

  const handleProviderChange = (newProvider: string) => {
    onChange({ provider: newProvider, model: '' })
  }

  const handleModelChange = (newModel: string) => {
    onChange({ ...value, model: newModel })
  }

  return (
    <div className="space-y-2">
      {label && (
        <label className="block text-sm font-medium text-surface-300">
          {label}
        </label>
      )}
      <div className="grid grid-cols-2 gap-2">
        {/* Provider Select */}
        <div>
          {loadingProviders ? (
            <div className="flex items-center gap-2 px-3 py-2 bg-surface-800 border border-surface-700 rounded-lg">
              <Loader2 className="w-4 h-4 animate-spin text-surface-500" />
              <span className="text-sm text-surface-500">Lade...</span>
            </div>
          ) : (
            <select
              value={value.provider}
              onChange={(e) => handleProviderChange(e.target.value)}
              disabled={disabled}
              className={clsx(
                'w-full input bg-surface-900/50 border-surface-700 focus:border-blue-500 text-sm',
                disabled && 'opacity-50 cursor-not-allowed'
              )}
            >
              <option value="">-- Provider --</option>
              {providers.map((p) => (
                <option key={p.name} value={p.name}>
                  {p.display_name || p.name}
                </option>
              ))}
            </select>
          )}
        </div>

        {/* Model Select */}
        <div>
          {loadingModels ? (
            <div className="flex items-center gap-2 px-3 py-2 bg-surface-800 border border-surface-700 rounded-lg">
              <Loader2 className="w-4 h-4 animate-spin text-surface-500" />
              <span className="text-sm text-surface-500">Modelle...</span>
            </div>
          ) : modelsError ? (
            <input
              type="text"
              value={value.model}
              onChange={(e) => handleModelChange(e.target.value)}
              placeholder="Modellname"
              disabled={disabled || !value.provider}
              className={clsx(
                'w-full input bg-surface-900/50 border-surface-700 focus:border-blue-500 font-mono text-sm',
                disabled && 'opacity-50 cursor-not-allowed'
              )}
            />
          ) : (
            <select
              value={value.model}
              onChange={(e) => handleModelChange(e.target.value)}
              disabled={disabled || !value.provider || models.length === 0}
              className={clsx(
                'w-full input bg-surface-900/50 border-surface-700 focus:border-blue-500 font-mono text-sm',
                disabled && 'opacity-50 cursor-not-allowed'
              )}
            >
              <option value="">-- Modell --</option>
              {models.map((m) => (
                <option key={m.id || m.name} value={m.name}>
                  {m.display_name || m.name}
                </option>
              ))}
            </select>
          )}
        </div>
      </div>
      {modelsError && (
        <p className="text-xs text-amber-400">{modelsError}</p>
      )}
    </div>
  )
}
