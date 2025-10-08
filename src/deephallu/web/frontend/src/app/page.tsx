"use client"

import { useState } from "react"
import { Header } from "@/components/Header"
import { Sidebar } from "@/components/Sidebar"
import { Toolbar } from "@/components/Toolbar"
import { OverviewView } from "@/components/views/OverviewView"
import { AttentionMapView } from "@/components/views/AttentionMapView"
import { TokenPredictionView } from "@/components/views/TokenPredictionView"
import { mockImages, runMockAnalysis } from "@/lib/mockData"
import { ViewType, AnalysisResult } from "@/lib/types"

export default function Home() {
  const [isSidebarOpen, setIsSidebarOpen] = useState(true)
  const [selectedImageId, setSelectedImageId] = useState<string>()
  const [selectedModel, setSelectedModel] = useState("llava-hf/llava-v1.6-mistral-7b-hf")
  const [currentView, setCurrentView] = useState<ViewType>('overview')
  const [isRunning, setIsRunning] = useState(false)
  const [analysisResult, setAnalysisResult] = useState<AnalysisResult | null>(null)
  const [selectedTokenId, setSelectedTokenId] = useState<number>()

  // Prompt and answer state
  const [prompt, setPrompt] = useState("Describe what you see in this image.")
  const [answer, setAnswer] = useState("")

  const toggleSidebar = () => {
    setIsSidebarOpen(!isSidebarOpen)
  }

  const handleSelectImage = (imageId: string) => {
    setSelectedImageId(imageId)

    // Reset analysis when new image is selected
    setAnalysisResult(null)
    setSelectedTokenId(undefined)

    // Set default prompt and answer from image questions if available
    const image = mockImages.find(img => img.id === imageId)
    if (image?.questions && image.questions.length > 0) {
      setPrompt(image.questions[0].question)
      setAnswer(image.questions[0].answer)
    } else {
      setPrompt("Describe what you see in this image.")
      setAnswer("")
    }
  }

  const handleRunAnalysis = async () => {
    if (!selectedImageId) return

    setIsRunning(true)
    try {
      const result = await runMockAnalysis(selectedImageId, selectedModel, prompt, answer)
      setAnalysisResult(result)
    } catch (error) {
      console.error('Analysis failed:', error)
    } finally {
      setIsRunning(false)
    }
  }

  const handleReset = () => {
    setAnalysisResult(null)
    setSelectedTokenId(undefined)
    setCurrentView('overview')
  }

  const selectedImage = selectedImageId ? mockImages.find(img => img.id === selectedImageId) || null : null

  return (
    <div className="h-screen flex flex-col bg-gray-50 dark:bg-gray-900">
      <Header onToggleSidebar={toggleSidebar} />

      <div className="flex flex-1 overflow-hidden">
        <Sidebar
          isOpen={isSidebarOpen}
          onClose={() => setIsSidebarOpen(false)}
          onSelectImage={handleSelectImage}
          selectedImageId={selectedImageId}
        />

        <div className="flex-1 flex flex-col overflow-hidden">
          <Toolbar
            selectedModel={selectedModel}
            onModelChange={setSelectedModel}
            onRunAnalysis={handleRunAnalysis}
            onReset={handleReset}
            isRunning={isRunning}
            hasImage={!!selectedImageId}
            currentView={currentView}
            onViewChange={setCurrentView}
            hasResults={!!analysisResult}
          />

          {/* Main Content Area */}
          <div className="flex-1 overflow-auto">
            {currentView === 'overview' && (
              <OverviewView
                selectedImage={selectedImage}
                analysisResult={analysisResult}
                prompt={prompt}
                setPrompt={setPrompt}
                answer={answer}
                setAnswer={setAnswer}
                onTokenClick={setSelectedTokenId}
                selectedTokenId={selectedTokenId}
              />
            )}
            {currentView === 'attention' && (
              <AttentionMapView
                selectedImage={selectedImage}
                analysisResult={analysisResult}
                selectedTokenId={selectedTokenId}
                onTokenClick={setSelectedTokenId}
              />
            )}
            {currentView === 'prediction' && (
              <TokenPredictionView
                analysisResult={analysisResult}
                selectedTokenId={selectedTokenId}
                onTokenClick={setSelectedTokenId}
              />
            )}
          </div>
        </div>
      </div>
    </div>
  )
}
