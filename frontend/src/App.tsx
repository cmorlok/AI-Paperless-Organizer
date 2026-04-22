import { Routes, Route } from 'react-router-dom'
import Layout from './components/Layout'
import AuthGuard from './components/AuthGuard'
import Dashboard from './components/Dashboard'
import CorrespondentManager from './components/CorrespondentManager'
import TagManager from './components/TagManager'
import TagCleanupWizard from './components/TagCleanupWizard'
import DocumentTypeManager from './components/DocumentTypeManager'
import SettingsPanel from './components/SettingsPanel'
import PromptEditor from './components/PromptEditor'
import DebugPanel from './components/DebugPanel'
import OcrManager from './components/OcrManager'
import CleanupManager from './components/CleanupManager'
import DocumentClassifier from './components/DocumentClassifier'
import RagChat from './pages/RagChat'
import KiLoesungen from './pages/KiLoesungen'
import CloudImport from './pages/CloudImport'
import DuplicateFinder from './pages/DuplicateFinder'
import LoginPage from './pages/LoginPage'
import SetupPage from './pages/SetupPage'

function App() {
  return (
    <Routes>
      {/* Public routes — no Layout, no AuthGuard */}
      <Route path="/login" element={<LoginPage />} />
      <Route path="/setup" element={<SetupPage />} />

      {/* Protected routes — AuthGuard wraps Layout + nested routes */}
      {/* NOTE: nested <Routes> is a React Router v6→v7 compatibility pattern. */}
      {/* If migrating to v7 data APIs, replace with <Outlet> inside Layout. */}
      <Route path="/*" element={
        <AuthGuard>
          <Layout>
            <Routes>
              <Route path="/" element={<Dashboard />} />
              <Route path="/classifier" element={<DocumentClassifier />} />
              <Route path="/ocr" element={<OcrManager />} />
              <Route path="/cleanup" element={<CleanupManager />} />
              <Route path="/correspondents" element={<CorrespondentManager />} />
              <Route path="/tags" element={<TagManager />} />
              <Route path="/tags/wizard" element={<TagCleanupWizard />} />
              <Route path="/document-types" element={<DocumentTypeManager />} />
              <Route path="/settings" element={<SettingsPanel />} />
              <Route path="/prompts" element={<PromptEditor />} />
              <Route path="/rag-chat" element={<RagChat />} />
              <Route path="/debug" element={<DebugPanel />} />
              <Route path="/ki-loesungen" element={<KiLoesungen />} />
              <Route path="/cloud-import" element={<CloudImport />} />
              <Route path="/duplicates" element={<DuplicateFinder />} />
            </Routes>
          </Layout>
        </AuthGuard>
      } />
    </Routes>
  )
}

export default App
