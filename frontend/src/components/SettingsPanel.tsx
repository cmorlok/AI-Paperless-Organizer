import { useState, useEffect } from 'react'
import { Save, Check, X, Eye, EyeOff, TestTube, Loader2, Cpu, Lock, Bug, Trash2, Ban, Key, Copy, Power, Code2 } from 'lucide-react'
import clsx from 'clsx'
import * as api from '../services/api'

interface LLMProvider {
  id: number
  name: string
  display_name: string
  api_key: string
  api_base_url: string
  is_active: boolean
  is_configured: boolean
}

export default function SettingsPanel() {
  // Paperless Settings
  const [paperlessUrl, setPaperlessUrl] = useState('')
  const [paperlessToken, setPaperlessToken] = useState('')
  const [showToken, setShowToken] = useState(false)
  const [paperlessSaving, setPaperlessSaving] = useState(false)
  const [paperlessTestResult, setPaperlessTestResult] = useState<{ success: boolean; message: string } | null>(null)

  // LLM Providers
  const [providers, setProviders] = useState<LLMProvider[]>([])
  // Dynamic provider/model state (LLM-09)
  const [availableProviders, setAvailableProviders] = useState<{name: string; display_name: string}[]>([])
  const [classifierModels, setClassifierModels] = useState<api.LLMModel[]>([])
  const [ocrModels, setOcrModels] = useState<api.LLMModel[]>([])

  // Unified LLM settings form state (LLM-09, D-06)
  const [classifierModel, setClassifierModel] = useState('')
  const [modelSaved, setModelSaved] = useState(false)
  const [ocrProvider, setOcrProvider] = useState('ollama')
  const [ocrModel, setOcrModel] = useState('')
  const [ocrModelSaved, setOcrModelSaved] = useState(false)
  const [selectedProviderApiKey, setSelectedProviderApiKey] = useState('')
  const [providerApiBaseUrl, setProviderApiBaseUrl] = useState('')
  const [connectionSaved, setConnectionSaved] = useState(false)
  
  // App Settings
  const [appSettings, setAppSettings] = useState({
    password_enabled: false,
    password_set: false,
    show_debug_menu: false,
    sidebar_compact: false,
    classifier_provider: 'ollama',
  })
  const [newPassword, setNewPassword] = useState('')
  const [showNewPassword, setShowNewPassword] = useState(false)
  const [savingAppSettings, setSavingAppSettings] = useState(false)
  const [appSettingsSaved, setAppSettingsSaved] = useState(false)
  
  // Ignored Items
  const [ignoredItems, setIgnoredItems] = useState<api.IgnoredItem[]>([])
  const [loadingIgnored, setLoadingIgnored] = useState(false)
  const [removingIgnored, setRemovingIgnored] = useState<number | null>(null)

  // API Keys
  const [apiKeys, setApiKeys] = useState<api.ApiKeyInfo[]>([])
  const [newKeyName, setNewKeyName] = useState('')
  const [generatedKey, setGeneratedKey] = useState<string | null>(null)
  const [generatingKey, setGeneratingKey] = useState(false)
  const [deletingKeyId, setDeletingKeyId] = useState<number | null>(null)

  useEffect(() => {
    loadSettings()
    loadAppSettings()
    loadIgnoredItems()
    loadApiKeys()
    loadProviders()  // NEW: load dynamic providers
  }, [])
  
  const loadIgnoredItems = async () => {
    setLoadingIgnored(true)
    try {
      const items = await api.getIgnoredItems()
      setIgnoredItems(items)
    } catch (e) {
      console.error('Failed to load ignored items:', e)
    } finally {
      setLoadingIgnored(false)
    }
  }
  
  const handleRemoveIgnored = async (id: number) => {
    setRemovingIgnored(id)
    try {
      await api.removeIgnoredItem(id)
      setIgnoredItems(ignoredItems.filter(item => item.id !== id))
    } catch (e) {
      console.error('Failed to remove ignored item:', e)
    } finally {
      setRemovingIgnored(null)
    }
  }

  const loadApiKeys = async () => {
    try {
      const keys = await api.listApiKeys()
      setApiKeys(keys)
    } catch (e) {
      console.error('Failed to load API keys:', e)
    }
  }

  const loadProviders = async () => {
    try {
      const providers = await api.getLLMProvidersDynamic()
      setAvailableProviders(providers)
    } catch (e) {
      console.error('Failed to load providers:', e)
    }
  }

  const loadClassifierModels = async (provider: string) => {
    try {
      const response = await api.getLLMProviderModels(provider)
      setClassifierModels(response.models)
      // If current model not in list, clear it
      if (classifierModel && !response.models.find(m => m.id === classifierModel)) {
        setClassifierModel('')
      }
    } catch (e) {
      console.error('Failed to load classifier models:', e)
      setClassifierModels([])
    }
  }

  const loadOcrModels = async (provider: string) => {
    try {
      const response = await api.getLLMProviderModels(provider)
      setOcrModels(response.models)
      // If current model not in list, clear it
      if (ocrModel && !response.models.find(m => m.id === ocrModel)) {
        setOcrModel('')
      }
    } catch (e) {
      console.error('Failed to load OCR models:', e)
      setOcrModels([])
    }
  }

  const handleGenerateKey = async () => {
    if (!newKeyName.trim()) return
    setGeneratingKey(true)
    try {
      const result = await api.generateApiKey(newKeyName.trim())
      setGeneratedKey(result.key)
      setNewKeyName('')
      loadApiKeys()
    } catch (e) {
      console.error('Failed to generate key:', e)
    } finally {
      setGeneratingKey(false)
    }
  }

  const handleDeleteKey = async (id: number) => {
    setDeletingKeyId(id)
    try {
      await api.deleteApiKey(id)
      loadApiKeys()
    } catch (e) {
      console.error('Failed to delete key:', e)
    } finally {
      setDeletingKeyId(null)
    }
  }

  const handleToggleKey = async (id: number) => {
    try {
      await api.toggleApiKey(id)
      loadApiKeys()
    } catch (e) {
      console.error('Failed to toggle key:', e)
    }
  }

  const loadAppSettings = async () => {
    try {
      const settings = await api.getAppSettings()
      setAppSettings(settings)
      // Load classifier_model and ocr_model for unified form (LLM-09)
      setClassifierModel((settings as any).classifier_model || '')
      setOcrModel((settings as any).ocr_model || '')
      // Load initial classifier models
      if (settings.classifier_provider) {
        loadClassifierModels(settings.classifier_provider)
      }
      // Load initial OCR models (always ollama for vision)
      loadOcrModels('ollama')
    } catch (e) {
      console.error('Error loading app settings:', e)
    }
  }

  const saveAppSettings = async (updates: Partial<typeof appSettings> & { password?: string }) => {
    setSavingAppSettings(true)
    try {
      await api.updateAppSettings(updates)
      await loadAppSettings()
      setAppSettingsSaved(true)
      setTimeout(() => setAppSettingsSaved(false), 2000)
      // Reload page to apply changes (like debug menu toggle)
      if (updates.show_debug_menu !== undefined) {
        window.location.reload()
      }
    } catch (e) {
      console.error('Error saving app settings:', e)
    } finally {
      setSavingAppSettings(false)
    }
  }

  const handleSetPassword = async () => {
    if (!newPassword) return
    await saveAppSettings({ password: newPassword, password_enabled: true })
    setNewPassword('')
  }

  const handleRemovePassword = async () => {
    await api.removePassword()
    await loadAppSettings()
    localStorage.removeItem('app_authenticated')
  }

  const loadSettings = async () => {
    try {
      const [paperlessSettings, llmProviders, appSettingsData] = await Promise.all([
        api.getPaperlessSettings(),
        api.getLLMProviders(),
        api.getAppSettings()
      ])

      setPaperlessUrl(paperlessSettings.url)
      setPaperlessToken(paperlessSettings.api_token)
      setProviders(llmProviders)
      setAppSettings(appSettingsData)

      // Load current LLM settings for unified form (LLM-09)
      const activeProvider = llmProviders.find((p: LLMProvider) => p.name === appSettingsData.classifier_provider)
      if (activeProvider) {
        setSelectedProviderApiKey(activeProvider.api_key === '***' ? '' : activeProvider.api_key)
        setProviderApiBaseUrl(activeProvider.api_base_url || '')
      }
    } catch (error) {
      console.error('Error loading settings:', error)
    }
  }

  const savePaperlessSettings = async () => {
    setPaperlessSaving(true)
    setPaperlessTestResult(null)
    try {
      await api.savePaperlessSettings({ url: paperlessUrl, api_token: paperlessToken })
      
      // Run detailed test via debug endpoint
      const testRes = await fetch('/api/debug/paperless-test', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ url: paperlessUrl, token: paperlessToken })
      })
      const testResult = await testRes.json()
      
      if (testResult.success && testResult.is_paperless) {
        setPaperlessTestResult({ 
          success: true, 
          message: `Verbindung erfolgreich! API gefunden: ${testResult.working_url}`
        })
      } else {
        // Fallback to simple status check
        const status = await api.getPaperlessStatus()
        if (status.connected) {
          setPaperlessTestResult({ success: true, message: 'Verbindung erfolgreich!' })
        } else {
          setPaperlessTestResult({ 
            success: false, 
            message: testResult.message || status.error || 'Verbindung fehlgeschlagen'
          })
        }
      }
    } catch (error) {
      setPaperlessTestResult({ success: false, message: 'Fehler beim Speichern' })
    } finally {
      setPaperlessSaving(false)
    }
  }

  return (
    <div className="space-y-8 max-w-4xl">
      {/* Paperless Settings */}
      <div className="card p-6">
        <h2 className="font-display font-semibold text-xl text-surface-100 mb-6">
          Paperless-ngx Verbindung
        </h2>

        <div className="space-y-4">
          <div>
            <label className="block text-sm font-medium text-surface-300 mb-2">
              Paperless URL
            </label>
            <input
              type="url"
              value={paperlessUrl}
              onChange={(e) => setPaperlessUrl(e.target.value)}
              placeholder="https://paperless.example.com"
              className="input"
            />
            <p className="mt-1 text-xs text-surface-500">
              Tipp: Für lokales Paperless nutze <code className="text-primary-400">http://host.docker.internal:PORT</code>
            </p>
          </div>

          <div>
            <label className="block text-sm font-medium text-surface-300 mb-2">
              API Token
            </label>
            <div className="relative">
              <input
                type={showToken ? 'text' : 'password'}
                value={paperlessToken}
                onChange={(e) => setPaperlessToken(e.target.value)}
                placeholder="Token aus Paperless Admin-Bereich"
                className="input pr-12"
              />
              <button
                type="button"
                onClick={() => setShowToken(!showToken)}
                className="absolute right-3 top-1/2 -translate-y-1/2 text-surface-400 hover:text-surface-200"
              >
                {showToken ? <EyeOff className="w-5 h-5" /> : <Eye className="w-5 h-5" />}
              </button>
            </div>
          </div>

          {paperlessTestResult && (
            <div className={clsx(
              'p-4 rounded-lg',
              paperlessTestResult.success 
                ? 'bg-emerald-500/10 border border-emerald-500/30'
                : 'bg-red-500/10 border border-red-500/30'
            )}>
              <div className="flex items-center gap-2">
                {paperlessTestResult.success ? (
                  <Check className="w-5 h-5 text-emerald-400" />
                ) : (
                  <X className="w-5 h-5 text-red-400" />
                )}
                <span className={paperlessTestResult.success ? 'text-emerald-400' : 'text-red-400'}>
                  {paperlessTestResult.success ? 'Verbindung erfolgreich!' : 'Verbindung fehlgeschlagen'}
                </span>
              </div>
              <p className="mt-2 text-sm text-surface-300">{paperlessTestResult.message}</p>
            </div>
          )}

          <button
            onClick={savePaperlessSettings}
            disabled={paperlessSaving}
            className="btn btn-primary flex items-center gap-2"
          >
            {paperlessSaving ? (
              <Loader2 className="w-4 h-4 animate-spin" />
            ) : (
              <Save className="w-4 h-4" />
            )}
            Speichern & Testen
          </button>
        </div>
      </div>

      {/* Unified LLM Settings Form (LLM-09, D-06) */}
      <div className="card p-6">
        <h2 className="font-display font-semibold text-xl text-surface-100 mb-6 flex items-center gap-2">
          <Cpu className="w-5 h-5 text-primary-400" />
          LLM Einstellungen
        </h2>

        {/* Section 1: Classifier LLM */}
        <div className="mb-8 space-y-4">
          <h3 className="text-lg font-medium text-surface-100 border-b border-surface-700 pb-2">
            Klassifizierung & Bereinigung
          </h3>

          {/* Provider dropdown */}
          <div>
            <label className="block text-sm font-medium text-surface-300 mb-2">Provider</label>
            <select
              value={appSettings.classifier_provider || 'ollama'}
              onChange={async (e) => {
                const providerName = e.target.value
                setAppSettings(prev => ({ ...prev, classifier_provider: providerName }))
                await api.updateAppSettings({ classifier_provider: providerName })
                // Update api_key/base_url state for newly selected provider
                const provider = providers.find((p: LLMProvider) => p.name === providerName)
                if (provider) {
                  setSelectedProviderApiKey(provider.api_key === '***' ? '' : provider.api_key)
                  setProviderApiBaseUrl(provider.api_base_url || '')
                }
                // Fetch models for new provider
                await loadClassifierModels(providerName)
              }}
              className="input w-full"
            >
              {availableProviders.map((p) => (
                <option key={p.name} value={p.name}>{p.display_name}</option>
              ))}
            </select>
          </div>

          {/* Show api_key for non-Ollama providers */}
          {appSettings.classifier_provider !== 'ollama' && (
            <div>
              <label className="block text-sm font-medium text-surface-300 mb-2">API Key</label>
              <input
                type="password"
                value={selectedProviderApiKey}
                onChange={(e) => setSelectedProviderApiKey(e.target.value)}
                placeholder="API Key eingeben"
                className="input w-full"
              />
            </div>
          )}

          {/* Model field — sets classifier_model in AppSettings */}
          <div>
            <label className="block text-sm font-medium text-surface-300 mb-2">Model</label>
            <select
              value={classifierModel}
              onChange={(e) => setClassifierModel(e.target.value)}
              className="input w-full"
            >
              <option value="">Model auswaehlen...</option>
              {classifierModels.map((m) => (
                <option key={m.id} value={m.id}>{m.display_name || m.name}</option>
              ))}
            </select>
            {classifierModel && (
              <button
                onClick={async () => {
                  await api.updateAppSettings({ classifier_model: classifierModel })
                  setModelSaved(true)
                  setTimeout(() => setModelSaved(false), 2000)
                }}
                className="btn btn-secondary mt-2"
              >
                Model speichern
              </button>
            )}
            {modelSaved && <span className="ml-2 text-emerald-400 text-sm">Gespeichert</span>}
          </div>

          {/* api_base_url — conditional display */}
          {(appSettings.classifier_provider === 'ollama' || appSettings.classifier_provider === 'azure') && (
            <div>
              <label className="block text-sm font-medium text-surface-300 mb-2">
                {appSettings.classifier_provider === 'ollama' ? 'Ollama URL' : 'Azure Endpoint'}
              </label>
              <input
                type="url"
                value={providerApiBaseUrl}
                onChange={(e) => setProviderApiBaseUrl(e.target.value)}
                placeholder={
                  appSettings.classifier_provider === 'ollama'
                    ? 'http://localhost:11434'
                    : 'https://xxx.openai.azure.com'
                }
                className="input w-full"
              />
            </div>
          )}
        </div>

        {/* Section 2: OCR LLM */}
        <div className="space-y-4">
          <h3 className="text-lg font-medium text-surface-100 border-b border-surface-700 pb-2">
            OCR (Texterkennung)
          </h3>

          {/* OCR Provider — always Ollama for vision */}
          <div>
            <label className="block text-sm font-medium text-surface-300 mb-2">OCR Provider</label>
            <select
              value={ocrProvider}
              onChange={async (e) => {
                const newProvider = e.target.value
                setOcrProvider(newProvider)
                await loadOcrModels(newProvider)  // Fetch models for OCR
              }}
              className="input w-full"
            >
              <option value="ollama">Ollama (Lokal)</option>
            </select>
            <p className="mt-1 text-xs text-surface-500">OCR verwendet Ollama Vision-Modelle</p>
          </div>

          {/* OCR Model — sets ocr_model in AppSettings */}
          <div>
            <label className="block text-sm font-medium text-surface-300 mb-2">OCR Model</label>
            <select
              value={ocrModel}
              onChange={(e) => setOcrModel(e.target.value)}
              className="input w-full"
            >
              <option value="">OCR Model auswaehlen...</option>
              {ocrModels.map((m) => (
                <option key={m.id} value={m.id}>{m.display_name || m.name}</option>
              ))}
            </select>
            {ocrModel && (
              <button
                onClick={async () => {
                  await api.updateAppSettings({ ocr_model: ocrModel })
                  setOcrModelSaved(true)
                  setTimeout(() => setOcrModelSaved(false), 2000)
                }}
                className="btn btn-secondary mt-2"
              >
                OCR Model speichern
              </button>
            )}
            {ocrModelSaved && <span className="ml-2 text-emerald-400 text-sm">Gespeichert</span>}
          </div>
        </div>

        {/* Save connection settings button */}
        <div className="mt-6 pt-4 border-t border-surface-700">
          <button
            onClick={async () => {
              const provider = providers.find((p: LLMProvider) => p.name === appSettings.classifier_provider)
              if (provider) {
                await api.updateLLMProviderConnection(provider.id, selectedProviderApiKey, providerApiBaseUrl)
                setConnectionSaved(true)
                setTimeout(() => setConnectionSaved(false), 2000)
              }
            }}
            className="btn btn-primary flex items-center gap-2"
          >
            <Save className="w-4 h-4" />
            Verbindung speichern
          </button>
          {connectionSaved && <span className="ml-3 text-emerald-400 text-sm">Verbindung gespeichert</span>}
        </div>
      </div>

      {/* App Settings */}
      <div className="card p-6">
        <h2 className="font-display font-semibold text-lg text-surface-100 mb-6 flex items-center gap-2">
          <Lock className="w-5 h-5 text-primary-400" />
          App-Einstellungen
        </h2>

        <div className="space-y-6">
          {/* Password Protection */}
          <div className="p-4 rounded-lg bg-surface-800/50 border border-surface-700">
            <h3 className="font-medium text-surface-100 mb-3">Passwort-Schutz</h3>
            <p className="text-sm text-surface-400 mb-4">
              Schütze die gesamte Anwendung mit einem Passwort. Ohne Passwort ist kein Zugriff möglich!
            </p>
            
            {appSettings.password_set ? (
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2 text-emerald-400">
                  <Check className="w-4 h-4" />
                  <span>Passwort ist gesetzt</span>
                </div>
                <button
                  onClick={handleRemovePassword}
                  className="btn btn-danger btn-sm flex items-center gap-2"
                >
                  <Trash2 className="w-4 h-4" />
                  Entfernen
                </button>
              </div>
            ) : (
              <div className="flex gap-3">
                <div className="flex-1 relative">
                  <input
                    type={showNewPassword ? 'text' : 'password'}
                    value={newPassword}
                    onChange={(e) => setNewPassword(e.target.value)}
                    placeholder="Neues Passwort"
                    className="input w-full pr-10"
                  />
                  <button
                    type="button"
                    onClick={() => setShowNewPassword(!showNewPassword)}
                    className="absolute right-3 top-1/2 -translate-y-1/2 text-surface-400 hover:text-surface-200"
                  >
                    {showNewPassword ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
                  </button>
                </div>
                <button
                  onClick={handleSetPassword}
                  disabled={!newPassword || savingAppSettings}
                  className="btn btn-primary flex items-center gap-2"
                >
                  <Lock className="w-4 h-4" />
                  Setzen
                </button>
              </div>
            )}
          </div>

          {/* Debug Menu Toggle */}
          <div className="p-4 rounded-lg bg-surface-800/50 border border-surface-700">
            <div className="flex items-center justify-between">
              <div>
                <h3 className="font-medium text-surface-100 flex items-center gap-2">
                  <Bug className="w-4 h-4 text-surface-400" />
                  Debug-Menü in Sidebar
                </h3>
                <p className="text-sm text-surface-400 mt-1">
                  Zeigt Netzwerk-Diagnose-Tools in der Navigation
                </p>
              </div>
              <button
                onClick={() => saveAppSettings({ show_debug_menu: !appSettings.show_debug_menu })}
                className={clsx(
                  'relative w-12 h-6 rounded-full transition-colors',
                  appSettings.show_debug_menu ? 'bg-primary-500' : 'bg-surface-600'
                )}
              >
                <span className={clsx(
                  'absolute top-1 w-4 h-4 rounded-full bg-white transition-all',
                  appSettings.show_debug_menu ? 'left-7' : 'left-1'
                )} />
              </button>
            </div>
          </div>

          {/* Save Confirmation */}
          {appSettingsSaved && (
            <div className="flex items-center gap-2 text-emerald-400 text-sm">
              <Check className="w-4 h-4" />
              Einstellungen gespeichert
            </div>
          )}
        </div>
      </div>
      
      {/* Ignored Items */}
      <div className="card p-6">
        <h2 className="font-display font-semibold text-lg text-surface-100 mb-6 flex items-center gap-2">
          <Ban className="w-5 h-5 text-red-400" />
          Ignorierte Einträge
          {ignoredItems.length > 0 && (
            <span className="text-sm font-normal text-surface-400">({ignoredItems.length})</span>
          )}
        </h2>
        
        <p className="text-sm text-surface-400 mb-4">
          Diese Einträge werden bei KI-Analysen nicht mehr vorgeschlagen. Du kannst sie hier wieder aktivieren.
        </p>
        
        {loadingIgnored ? (
          <div className="flex items-center justify-center py-8">
            <Loader2 className="w-6 h-6 animate-spin text-primary-400" />
          </div>
        ) : ignoredItems.length === 0 ? (
          <div className="text-center py-8 text-surface-500">
            Keine ignorierten Einträge vorhanden.
          </div>
        ) : (
          <div className="space-y-2 max-h-80 overflow-y-auto">
            {ignoredItems.map(item => (
              <div 
                key={item.id} 
                className="flex items-center justify-between p-3 rounded-lg bg-surface-800/50 border border-surface-700"
              >
                <div className="flex-1">
                  <div className="flex items-center gap-2">
                    <span className="font-medium text-surface-200">{item.item_name}</span>
                    <span className="text-xs px-2 py-0.5 rounded bg-surface-700 text-surface-400">
                      {item.entity_type === 'tag' ? 'Tag' : 
                       item.entity_type === 'correspondent' ? 'Korrespondent' : 'Dokumententyp'}
                    </span>
                    <span className="text-xs px-2 py-0.5 rounded bg-surface-700 text-surface-400">
                      {item.analysis_type === 'nonsense' ? 'Unsinnig' :
                       item.analysis_type === 'correspondent_match' ? 'Korrespondent-Match' :
                       item.analysis_type === 'doctype_match' ? 'Dokumententyp-Match' : 'Ähnlich'}
                    </span>
                  </div>
                  <p className="text-xs text-surface-500 mt-1">
                    Ignoriert am {new Date(item.created_at).toLocaleDateString('de-DE')}
                  </p>
                </div>
                <button
                  onClick={() => handleRemoveIgnored(item.id)}
                  disabled={removingIgnored === item.id}
                  className="p-2 rounded text-surface-400 hover:text-emerald-400 hover:bg-emerald-500/10 transition-colors"
                  title="Wieder aktivieren"
                >
                  {removingIgnored === item.id ? (
                    <Loader2 className="w-4 h-4 animate-spin" />
                  ) : (
                    <Check className="w-4 h-4" />
                  )}
                </button>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Externe API */}
      <div className="card p-6">
        <div className="flex items-center gap-3 mb-4">
          <Code2 className="w-6 h-6 text-primary-400" />
          <h2 className="text-xl font-semibold text-surface-100">Externe API (RAG Chat)</h2>
        </div>

        <p className="text-sm text-surface-400 mb-4">
          Binde den Dokumenten-Chat in deine eigene Software ein. Generiere einen API-Key und nutze die REST-API,
          um per Chat Dokumente zu finden und Fragen zu beantworten.
        </p>

        {/* API Key generieren */}
        <div className="border border-surface-600 rounded-lg p-4 mb-4">
          <h3 className="text-sm font-semibold text-surface-200 mb-3 flex items-center gap-2">
            <Key className="w-4 h-4 text-primary-400" />
            API-Keys verwalten
          </h3>

          <div className="flex gap-2 mb-3">
            <input
              type="text"
              value={newKeyName}
              onChange={(e) => setNewKeyName(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && handleGenerateKey()}
              placeholder="Name fuer den Key (z.B. 'Mein ERP-System')"
              className="flex-1 px-3 py-2 bg-surface-700 border border-surface-600 rounded-lg text-sm text-surface-100 placeholder-surface-500"
            />
            <button
              onClick={handleGenerateKey}
              disabled={!newKeyName.trim() || generatingKey}
              className="btn btn-primary flex items-center gap-2"
            >
              {generatingKey ? <Loader2 className="w-4 h-4 animate-spin" /> : <Key className="w-4 h-4" />}
              Key generieren
            </button>
          </div>

          {generatedKey && (
            <div className="bg-green-900/30 border border-green-700 rounded-lg p-3 mb-3">
              <div className="text-sm font-medium text-green-300 mb-1">Neuer API-Key erstellt:</div>
              <div className="flex items-center gap-2">
                <code className="flex-1 bg-surface-800 px-3 py-2 rounded border border-green-700 text-sm font-mono text-green-300 break-all">
                  {generatedKey}
                </code>
                <button
                  onClick={() => { navigator.clipboard.writeText(generatedKey); }}
                  className="p-2 text-green-400 hover:bg-green-900/50 rounded"
                  title="Kopieren"
                >
                  <Copy className="w-4 h-4" />
                </button>
              </div>
              <p className="text-xs text-green-400 mt-1">
                Speichere diesen Key jetzt! Er wird nicht erneut angezeigt.
              </p>
            </div>
          )}

          {apiKeys.length > 0 && (
            <div className="space-y-2">
              {apiKeys.map((k) => (
                <div key={k.id} className="flex items-center justify-between bg-surface-700/50 rounded-lg px-3 py-2">
                  <div className="flex items-center gap-3">
                    <Key className={clsx('w-4 h-4', k.is_active ? 'text-green-400' : 'text-surface-500')} />
                    <div>
                      <div className="text-sm font-medium text-surface-100">{k.name}</div>
                      <div className="text-xs text-surface-400">
                        {k.key_prefix}... | Erstellt: {k.created_at ? new Date(k.created_at).toLocaleDateString('de-DE') : '-'}
                        {k.last_used_at && <> | Zuletzt: {new Date(k.last_used_at).toLocaleDateString('de-DE')}</>}
                      </div>
                    </div>
                  </div>
                  <div className="flex items-center gap-1">
                    <button
                      onClick={() => handleToggleKey(k.id)}
                      className={clsx('p-1.5 rounded', k.is_active ? 'text-green-400 hover:bg-surface-600' : 'text-surface-500 hover:bg-surface-600')}
                      title={k.is_active ? 'Deaktivieren' : 'Aktivieren'}
                    >
                      <Power className="w-4 h-4" />
                    </button>
                    <button
                      onClick={() => handleDeleteKey(k.id)}
                      disabled={deletingKeyId === k.id}
                      className="p-1.5 text-red-400 hover:text-red-300 hover:bg-surface-600 rounded"
                      title="Loeschen"
                    >
                      <Trash2 className="w-4 h-4" />
                    </button>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>

        {/* API Dokumentation */}
        <div className="border border-surface-600 rounded-lg p-4">
          <div className="flex items-center justify-between mb-3">
            <h3 className="text-sm font-semibold text-surface-200 flex items-center gap-2">
              <Code2 className="w-4 h-4 text-primary-400" />
              API-Dokumentation
            </h3>
            <a
              href="/docs"
              target="_blank"
              rel="noopener"
              className="btn btn-primary flex items-center gap-2 text-sm"
            >
              <TestTube className="w-4 h-4" />
              Swagger UI oeffnen
            </a>
          </div>

          <div className="space-y-4 text-sm">
            <div>
              <h4 className="font-medium text-surface-100 mb-1">Was kann die API?</h4>
              <ul className="list-disc list-inside text-surface-400 space-y-1">
                <li><strong className="text-surface-200">Chat</strong> - Stelle Fragen zu deinen Dokumenten, die KI antwortet mit Quellenangaben</li>
                <li><strong className="text-surface-200">Suche</strong> - Semantische Dokumentensuche (findet Bedeutung, nicht nur Stichworte)</li>
                <li><strong className="text-surface-200">Sessions</strong> - Chat-Verlaeufe werden gespeichert und koennen abgerufen werden</li>
              </ul>
            </div>

            <div>
              <h4 className="font-medium text-surface-100 mb-1">Authentifizierung</h4>
              <p className="text-surface-400 mb-2">Sende den API-Key als Bearer-Token im Authorization-Header:</p>
              <pre className="bg-surface-900 text-green-400 rounded-lg p-3 overflow-x-auto text-xs">
{`Authorization: Bearer po_dein_api_key_hier`}</pre>
            </div>

            <div>
              <h4 className="font-medium text-surface-100 mb-1">Chat-Endpunkt (SSE Streaming)</h4>
              <pre className="bg-surface-900 text-green-400 rounded-lg p-3 overflow-x-auto text-xs">
{`POST /api/rag/chat
Content-Type: application/json
Authorization: Bearer po_dein_api_key

{
  "question": "Welche Rechnungen habe ich von Vodafone?",
  "session_id": null,
  "filters": {
    "correspondent_id": null,
    "document_type_id": null,
    "tags": null
  }
}`}</pre>
              <p className="text-surface-500 mt-1">Antwort: Server-Sent Events (SSE) mit Token-Stream und Quellenangaben.</p>
            </div>

            <div>
              <h4 className="font-medium text-surface-100 mb-1">Such-Endpunkt</h4>
              <pre className="bg-surface-900 text-green-400 rounded-lg p-3 overflow-x-auto text-xs">
{`POST /api/rag/search
Content-Type: application/json
Authorization: Bearer po_dein_api_key

{
  "query": "Mietvertrag",
  "limit": 5
}`}</pre>
              <p className="text-surface-500 mt-1">Antwort: JSON mit relevanten Dokumenten, Scores und Snippets.</p>
            </div>

            <div>
              <h4 className="font-medium text-surface-100 mb-1">Weitere Endpunkte</h4>
              <div className="bg-surface-700/50 rounded-lg p-3 space-y-1 text-xs font-mono text-surface-200">
                <div><span className="text-blue-400">GET</span> /api/rag/sessions - Alle Chat-Sessions auflisten</div>
                <div><span className="text-blue-400">GET</span> /api/rag/sessions/:id - Einzelne Session mit Nachrichten</div>
                <div><span className="text-red-400">DELETE</span> /api/rag/sessions/:id - Session loeschen</div>
                <div><span className="text-green-400">POST</span> /api/rag/index/start - Indexierung starten</div>
                <div><span className="text-blue-400">GET</span> /api/rag/index/status - Indexierungsstatus</div>
                <div><span className="text-blue-400">GET</span> /api/rag/config - RAG-Konfiguration</div>
              </div>
            </div>

            <div className="bg-primary-900/30 border border-primary-700 rounded-lg p-3">
              <p className="text-primary-300 text-xs">
                <strong>Tipp:</strong> In der Swagger UI kannst du alle Endpunkte direkt im Browser testen.
                Klicke auf einen Endpunkt, dann auf "Try it out", fuege deinen API-Key ein und sende die Anfrage.
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
