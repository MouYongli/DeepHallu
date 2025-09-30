"use client"

import { useState } from "react"
import {
  Play,
  Pause,
  RotateCcw,
  Download,
  Eye,
  Settings,
  Zap,
  Image as ImageIcon,
  MessageSquare,
  Activity,
  AlertTriangle,
  CheckCircle
} from "lucide-react"
import { Dialog, DialogContent, DialogTrigger } from "@radix-ui/react-dialog"
import { Separator } from "@radix-ui/react-separator"
import { cn } from "@/lib/utils"
import { runMockAnalysis, modelConfigurations, MockAnalysisResult, mockImages } from "@/lib/mockData"

interface MainContentProps {
  selectedImageId?: string
}

export function MainContent({ selectedImageId }: MainContentProps) {
  const [isRunning, setIsRunning] = useState(false)
  const [selectedModel, setSelectedModel] = useState("llava-next-7b")
  const [prompt, setPrompt] = useState("Describe what you see in this image.")
  const [showAttentionMap, setShowAttentionMap] = useState(false)
  const [analysisResult, setAnalysisResult] = useState<MockAnalysisResult | null>(null)
  const [hoveredToken, setHoveredToken] = useState<string | null>(null)

  const selectedImage = selectedImageId ? mockImages.find(img => img.id === selectedImageId) : null

  const handleRunModel = async () => {
    setIsRunning(true)
    try {
      const result = await runMockAnalysis(prompt, selectedModel, selectedImageId)
      setAnalysisResult(result)
    } catch (error) {
      console.error('Analysis failed:', error)
    } finally {
      setIsRunning(false)
    }
  }

  const handleReset = () => {
    setAnalysisResult(null)
    setPrompt("Describe what you see in this image.")
    setShowAttentionMap(false)
  }

  const currentModelConfig = modelConfigurations.find(m => m.id === selectedModel)

  // Generate attention patches based on hovered token
  const getAttentionPatches = () => {
    if (!analysisResult || !hoveredToken || !showAttentionMap) return []

    const tokenData = analysisResult.attentionData.find(data => data.token === hoveredToken)
    if (!tokenData || tokenData.type !== 'image') return []

    // Create patches around the attention point
    const patches = []
    const [x, y] = tokenData.position
    const weight = tokenData.weight
    const patchSize = 15 + (weight * 25) // Size based on attention weight

    // Use token as seed for consistent randomness
    const seed = tokenData.token.split('').reduce((a, b) => {
      a = ((a << 5) - a) + b.charCodeAt(0)
      return a & a
    }, 0)

    const seededRandom = (n: number) => {
      const x = Math.sin(seed + n) * 10000
      return x - Math.floor(x)
    }

    // Main attention patch
    patches.push({
      x: Math.max(0, x - patchSize/2),
      y: Math.max(0, y - patchSize/2),
      width: patchSize,
      height: patchSize,
      opacity: weight * 0.8,
      color: '#3B82F6' // Blue
    })

    // Add some surrounding patches for visual effect
    for (let i = 0; i < 4; i++) {
      const offsetX = (seededRandom(i * 2) - 0.5) * 50
      const offsetY = (seededRandom(i * 2 + 1) - 0.5) * 50
      const smallSize = patchSize * (0.4 + seededRandom(i + 10) * 0.4)

      patches.push({
        x: Math.max(0, x + offsetX - smallSize/2),
        y: Math.max(0, y + offsetY - smallSize/2),
        width: smallSize,
        height: smallSize,
        opacity: weight * 0.4 * seededRandom(i + 20),
        color: i % 2 === 0 ? '#60A5FA' : '#93C5FD' // Different blues
      })
    }

    return patches
  }

  return (
    <div className="flex-1 flex flex-col h-full bg-gray-50 dark:bg-gray-900">
      {/* Main Controls */}
      <div className="bg-white dark:bg-gray-800 border-b border-gray-200 dark:border-gray-700 p-4">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-xl font-semibold text-gray-900 dark:text-white">
            LLaVA-Next Token Analysis
          </h2>
          <div className="flex items-center space-x-2">
            <button
              onClick={() => setShowAttentionMap(!showAttentionMap)}
              className={cn(
                "flex items-center space-x-2 px-3 py-2 rounded-md text-sm font-medium transition-colors",
                showAttentionMap
                  ? "bg-blue-100 text-blue-700 dark:bg-blue-900 dark:text-blue-300"
                  : "bg-gray-100 text-gray-700 dark:bg-gray-700 dark:text-gray-300 hover:bg-gray-200 dark:hover:bg-gray-600"
              )}
            >
              <Eye className="h-4 w-4" />
              <span>Attention Map</span>
            </button>
            <button className="flex items-center space-x-2 px-3 py-2 bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300 rounded-md text-sm font-medium hover:bg-gray-200 dark:hover:bg-gray-600 transition-colors">
              <Settings className="h-4 w-4" />
              <span>Settings</span>
            </button>
          </div>
        </div>

        {/* Model Selection */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
              Model
            </label>
            <select
              value={selectedModel}
              onChange={(e) => setSelectedModel(e.target.value)}
              className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-700 text-gray-900 dark:text-white focus:ring-2 focus:ring-blue-500 focus:border-transparent"
            >
              {modelConfigurations.map((model) => (
                <option key={model.id} value={model.id}>
                  {model.name} ({model.parameters})
                </option>
              ))}
            </select>
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
              Prompt
            </label>
            <input
              type="text"
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-700 text-gray-900 dark:text-white focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              placeholder="Enter your prompt..."
            />
          </div>
        </div>

        {/* Action Buttons */}
        <div className="flex items-center space-x-3 mt-4">
          <button
            onClick={handleRunModel}
            disabled={isRunning || !selectedImageId}
            className="flex items-center space-x-2 px-4 py-2 bg-blue-600 hover:bg-blue-700 disabled:bg-blue-400 text-white rounded-md font-medium transition-colors"
          >
            {isRunning ? (
              <>
                <Pause className="h-4 w-4" />
                <span>Running...</span>
              </>
            ) : (
              <>
                <Play className="h-4 w-4" />
                <span>Run Analysis</span>
              </>
            )}
          </button>

          <button
            onClick={handleReset}
            className="flex items-center space-x-2 px-4 py-2 bg-gray-200 dark:bg-gray-700 text-gray-700 dark:text-gray-300 rounded-md font-medium hover:bg-gray-300 dark:hover:bg-gray-600 transition-colors"
          >
            <RotateCcw className="h-4 w-4" />
            <span>Reset</span>
          </button>

          <button className="flex items-center space-x-2 px-4 py-2 bg-green-600 hover:bg-green-700 text-white rounded-md font-medium transition-colors">
            <Download className="h-4 w-4" />
            <span>Export</span>
          </button>
        </div>
      </div>

      {/* Content Area */}
      <div className="flex-1 grid grid-cols-1 lg:grid-cols-2 gap-6 p-6">
        {/* Image Display */}
        <div className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-4">
          <div className="flex items-center space-x-2 mb-4">
            <ImageIcon className="h-5 w-5 text-gray-600 dark:text-gray-400" />
            <h3 className="text-lg font-medium text-gray-900 dark:text-white">
              Input Image
            </h3>
          </div>

          <div className="w-full bg-gray-100 dark:bg-gray-700 rounded-lg border-2 border-dashed border-gray-300 dark:border-gray-600 flex items-center justify-center min-h-[400px]">
            {selectedImage ? (
              <div className="text-center w-full">
                {(selectedImage.path.startsWith('/examples/') || selectedImage.path.startsWith('blob:')) ? (
                  <div className="relative inline-block">
                    <img
                      src={selectedImage.path}
                      alt={selectedImage.description}
                      className="max-w-full max-h-[350px] object-contain rounded-lg mb-2"
                      onError={(e) => {
                        const target = e.target as HTMLImageElement;
                        target.style.display = 'none';
                        target.nextElementSibling?.classList.remove('hidden');
                      }}
                    />
                    {/* Attention overlay */}
                    <div className="absolute inset-0 pointer-events-none">
                      <svg className="w-full h-full" style={{ position: 'absolute', top: 0, left: 0 }}>
                        {getAttentionPatches().map((patch, index) => (
                          <rect
                            key={index}
                            x={patch.x}
                            y={patch.y}
                            width={patch.width}
                            height={patch.height}
                            fill={patch.color}
                            opacity={patch.opacity}
                            rx="4"
                            className="animate-pulse"
                          />
                        ))}
                      </svg>
                    </div>
                  </div>
                ) : null}
                <div className={cn(
                  "w-80 h-80 bg-gradient-to-br from-blue-100 to-green-100 dark:from-blue-800 dark:to-green-800 rounded-lg flex items-center justify-center mb-2 mx-auto",
                  (selectedImage.path.startsWith('/examples/') || selectedImage.path.startsWith('blob:')) ? 'hidden' : ''
                )}>
                  <ImageIcon className="h-24 w-24 text-blue-600 dark:text-blue-400" />
                </div>
                <p className="text-sm text-gray-700 dark:text-gray-300 font-medium">
                  {selectedImage.name}
                </p>
                <p className="text-xs text-gray-500 dark:text-gray-400">
                  {selectedImage.description}
                </p>
              </div>
            ) : (
              <div className="text-center">
                <ImageIcon className="h-12 w-12 text-gray-400 mx-auto mb-2" />
                <p className="text-gray-500 dark:text-gray-400">
                  No image selected
                </p>
                <p className="text-sm text-gray-400 dark:text-gray-500">
                  Upload or select an image from the sidebar
                </p>
              </div>
            )}
          </div>

          {/* Image Info */}
          {selectedImage && (
            <div className="mt-4 grid grid-cols-2 gap-4 text-sm">
              <div>
                <span className="text-gray-500 dark:text-gray-400">Resolution:</span>
                <span className="ml-2 text-gray-900 dark:text-white">{selectedImage.resolution}</span>
              </div>
              <div>
                <span className="text-gray-500 dark:text-gray-400">Format:</span>
                <span className="ml-2 text-gray-900 dark:text-white">{selectedImage.format}</span>
              </div>
              <div>
                <span className="text-gray-500 dark:text-gray-400">Size:</span>
                <span className="ml-2 text-gray-900 dark:text-white">{selectedImage.size}</span>
              </div>
              <div>
                <span className="text-gray-500 dark:text-gray-400">Dataset:</span>
                <span className="ml-2 text-gray-900 dark:text-white">{selectedImage.dataset}</span>
              </div>
              <div className="col-span-2">
                <span className="text-gray-500 dark:text-gray-400">Category:</span>
                <span className="ml-2 text-gray-900 dark:text-white">{selectedImage.category}</span>
              </div>
            </div>
          )}
        </div>

        {/* Results & Analysis */}
        <div className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-4">
          <div className="flex items-center space-x-2 mb-4">
            <Activity className="h-5 w-5 text-gray-600 dark:text-gray-400" />
            <h3 className="text-lg font-medium text-gray-900 dark:text-white">
              Token Analysis
            </h3>
          </div>

          {isRunning ? (
            <div className="space-y-4">
              <div className="animate-pulse">
                <div className="h-4 bg-gray-200 dark:bg-gray-700 rounded w-3/4 mb-2"></div>
                <div className="h-4 bg-gray-200 dark:bg-gray-700 rounded w-1/2 mb-2"></div>
                <div className="h-4 bg-gray-200 dark:bg-gray-700 rounded w-5/6"></div>
              </div>
              <div className="flex items-center space-x-2 text-blue-600 dark:text-blue-400">
                <Zap className="h-4 w-4 animate-spin" />
                <span className="text-sm">Processing attention patterns...</span>
              </div>
            </div>
          ) : (
            <div className="space-y-4">
              {/* Model Response */}
              <div>
                <h4 className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                  Model Response
                </h4>
                <div className="bg-gray-50 dark:bg-gray-700 rounded-lg p-3 min-h-[100px]">
                  {analysisResult ? (
                    <div className="space-y-3">
                      <p className="text-gray-900 dark:text-white text-sm">
                        {analysisResult.response}
                      </p>
                      <div className="flex items-center justify-between text-xs">
                        <div className="flex items-center space-x-2">
                          <span className="text-gray-500">Confidence:</span>
                          <span className={cn(
                            "font-medium",
                            analysisResult.confidence > 0.8 ? "text-green-600" :
                            analysisResult.confidence > 0.6 ? "text-yellow-600" : "text-red-600"
                          )}>
                            {(analysisResult.confidence * 100).toFixed(1)}%
                          </span>
                        </div>
                        <div className="flex items-center space-x-2">
                          <span className="text-gray-500">Hallucination Risk:</span>
                          <span className={cn(
                            "font-medium flex items-center space-x-1",
                            analysisResult.hallucinationScore < 0.2 ? "text-green-600" :
                            analysisResult.hallucinationScore < 0.4 ? "text-yellow-600" : "text-red-600"
                          )}>
                            {analysisResult.hallucinationScore < 0.2 ? (
                              <CheckCircle className="h-3 w-3" />
                            ) : (
                              <AlertTriangle className="h-3 w-3" />
                            )}
                            <span>{(analysisResult.hallucinationScore * 100).toFixed(1)}%</span>
                          </span>
                        </div>
                      </div>
                    </div>
                  ) : (
                    <p className="text-gray-600 dark:text-gray-400 text-sm italic">
                      Run analysis to see model response...
                    </p>
                  )}
                </div>
              </div>

              <Separator className="bg-gray-200 dark:bg-gray-700 h-px" />

              {/* Token Statistics */}
              <div>
                <h4 className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                  Token Statistics
                </h4>
                <div className="grid grid-cols-2 gap-4 text-sm">
                  <div className="bg-blue-50 dark:bg-blue-900/20 rounded-lg p-3">
                    <div className="text-blue-700 dark:text-blue-300 font-medium">Image Tokens</div>
                    <div className="text-2xl font-bold text-blue-900 dark:text-blue-100">
                      {analysisResult?.imageTokens || '--'}
                    </div>
                  </div>
                  <div className="bg-green-50 dark:bg-green-900/20 rounded-lg p-3">
                    <div className="text-green-700 dark:text-green-300 font-medium">Text Tokens</div>
                    <div className="text-2xl font-bold text-green-900 dark:text-green-100">
                      {analysisResult?.textTokens || '--'}
                    </div>
                  </div>
                </div>
              </div>

              {/* Model Performance */}
              {currentModelConfig && (
                <div>
                  <h4 className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                    Model Performance
                  </h4>
                  <div className="grid grid-cols-2 gap-4 text-sm">
                    <div className="bg-purple-50 dark:bg-purple-900/20 rounded-lg p-3">
                      <div className="text-purple-700 dark:text-purple-300 font-medium">Accuracy</div>
                      <div className="text-2xl font-bold text-purple-900 dark:text-purple-100">
                        {currentModelConfig.accuracy}
                      </div>
                    </div>
                    <div className="bg-orange-50 dark:bg-orange-900/20 rounded-lg p-3">
                      <div className="text-orange-700 dark:text-orange-300 font-medium">Inference Time</div>
                      <div className="text-2xl font-bold text-orange-900 dark:text-orange-100">
                        {currentModelConfig.inferenceTime}
                      </div>
                    </div>
                  </div>
                </div>
              )}

              {/* Attention Visualization */}
              <div>
                <h4 className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                  Attention Patterns
                </h4>
                <div className="bg-gray-50 dark:bg-gray-700 rounded-lg p-6 text-center min-h-[150px] flex items-center justify-center">
                  {analysisResult && showAttentionMap ? (
                    <div className="w-full">
                      <div className="mb-4 text-xs text-gray-600 dark:text-gray-400">
                        Hover over tokens to see attention on image
                      </div>
                      <div className="grid grid-cols-3 gap-2 text-xs">
                        {analysisResult.attentionData.slice(0, 9).map((data, index) => (
                          <div
                            key={index}
                            className={cn(
                              "p-2 rounded border cursor-pointer transition-all duration-200",
                              data.type === 'image'
                                ? "bg-blue-100 border-blue-300 hover:bg-blue-200 hover:border-blue-400"
                                : "bg-green-100 border-green-300 hover:bg-green-200 hover:border-green-400",
                              hoveredToken === data.token && "ring-2 ring-blue-500 bg-blue-200 border-blue-400"
                            )}
                            onMouseEnter={() => setHoveredToken(data.token)}
                            onMouseLeave={() => setHoveredToken(null)}
                          >
                            <div className="font-medium">{data.token}</div>
                            <div className="text-gray-600">
                              Weight: {data.weight.toFixed(2)}
                            </div>
                            {data.type === 'image' && (
                              <div className="text-xs text-blue-600 mt-1">
                                → Image region
                              </div>
                            )}
                          </div>
                        ))}
                      </div>
                    </div>
                  ) : (
                    <div className="text-gray-500 dark:text-gray-400">
                      <Activity className="h-8 w-8 mx-auto mb-2" />
                      <p className="text-sm">
                        {analysisResult ? "Enable attention map to view patterns" : "Attention visualization will appear here"}
                      </p>
                    </div>
                  )}
                </div>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Status Bar */}
      <div className="bg-white dark:bg-gray-800 border-t border-gray-200 dark:border-gray-700 px-6 py-2">
        <div className="flex items-center justify-between text-sm">
          <div className="flex items-center space-x-4 text-gray-600 dark:text-gray-400">
            <span>Status: {isRunning ? 'Processing...' : analysisResult ? 'Complete' : 'Ready'}</span>
            <span>•</span>
            <span>Model: {currentModelConfig?.name}</span>
            {currentModelConfig && (
              <>
                <span>•</span>
                <span>Memory: {currentModelConfig.memoryUsage}</span>
              </>
            )}
          </div>
          <div className="flex items-center space-x-2 text-gray-600 dark:text-gray-400">
            <MessageSquare className="h-4 w-4" />
            <span>{analysisResult ? `Analysis completed at ${analysisResult.timestamp.toLocaleTimeString()}` : 'Ready for analysis'}</span>
          </div>
        </div>
      </div>
    </div>
  )
}