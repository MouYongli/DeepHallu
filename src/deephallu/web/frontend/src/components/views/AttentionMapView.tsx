"use client"

import { useState, useRef, useEffect } from "react"
import { ImageMetadata, AnalysisResult } from "@/lib/types"
import { cn } from "@/lib/utils"
import { Slider } from "@radix-ui/react-slider"

interface AttentionMapViewProps {
  selectedImage: ImageMetadata | null
  analysisResult: AnalysisResult | null
  selectedTokenId?: number
  onTokenClick?: (tokenId: number) => void
}

export function AttentionMapView({
  selectedImage,
  analysisResult,
  selectedTokenId,
  onTokenClick
}: AttentionMapViewProps) {
  const [hoveredTokenId, setHoveredTokenId] = useState<number | null>(null)
  const [threshold, setThreshold] = useState([0.0])
  const [focusMode, setFocusMode] = useState(false)
  const [showTopK, setShowTopK] = useState(20)
  const paddedCanvasRef = useRef<HTMLCanvasElement>(null)
  const unpaddedCanvasRef = useRef<HTMLCanvasElement>(null)

  const activeTokenId = hoveredTokenId ?? selectedTokenId

  // Get attention data for the active token
  const attentionData = analysisResult?.attention_map?.attn.find(
    a => a.token_id === activeTokenId
  )

  // Draw attention heatmap on canvas
  useEffect(() => {
    if (!selectedImage || !analysisResult?.attention_map) return

    // Draw padded attention
    const paddedCanvas = paddedCanvasRef.current
    if (paddedCanvas) {
      const ctx = paddedCanvas.getContext('2d')
      if (ctx) {
        const img = new Image()
        img.src = selectedImage.path
        img.onload = () => {
          ctx.clearRect(0, 0, paddedCanvas.width, paddedCanvas.height)
          ctx.drawImage(img, 0, 0, paddedCanvas.width, paddedCanvas.height)

          // Draw heatmap only if we have attention data
          if (attentionData) {
            const gridH = analysisResult.attention_map!.image_features.padded.grid_h
            const gridW = analysisResult.attention_map!.image_features.padded.grid_w
            const cellW = paddedCanvas.width / gridW
            const cellH = paddedCanvas.height / gridH

            const weights = attentionData.to_image_padded

            // Get top-K patches with their indices and weights
            const weightedPatches = weights
              .map((weight, idx) => ({ weight, idx }))
              .filter(p => p.weight >= threshold[0])
              .sort((a, b) => b.weight - a.weight)
              .slice(0, showTopK)

            const maxWeight = weightedPatches.length > 0 ? weightedPatches[0].weight : 1

            weightedPatches.forEach(({ weight, idx }, rank) => {
              const row = Math.floor(idx / gridW)
              const col = idx % gridW
              const x = col * cellW
              const y = row * cellH

              const normalizedWeight = focusMode ? weight / maxWeight : weight
              const opacity = Math.min(0.3 + normalizedWeight * 0.7, 0.9)

              // Draw filled rectangle
              ctx.fillStyle = `rgba(59, 130, 246, ${opacity})`
              ctx.fillRect(x, y, cellW, cellH)

              // Draw border
              ctx.strokeStyle = `rgba(255, 255, 255, 0.8)`
              ctx.lineWidth = 1
              ctx.strokeRect(x, y, cellW, cellH)

              // Draw weight value
              ctx.fillStyle = 'white'
              ctx.font = 'bold 10px monospace'
              ctx.textAlign = 'center'
              ctx.textBaseline = 'middle'
              ctx.shadowColor = 'black'
              ctx.shadowBlur = 3
              const text = weight.toFixed(3)
              ctx.fillText(text, x + cellW / 2, y + cellH / 2)
              ctx.shadowBlur = 0
            })
          }
        }
      }
    }

    // Draw unpadded attention
    const unpaddedCanvas = unpaddedCanvasRef.current
    if (unpaddedCanvas) {
      const ctx = unpaddedCanvas.getContext('2d')
      if (ctx) {
        const img = new Image()
        img.src = selectedImage.path
        img.onload = () => {
          ctx.clearRect(0, 0, unpaddedCanvas.width, unpaddedCanvas.height)
          ctx.drawImage(img, 0, 0, unpaddedCanvas.width, unpaddedCanvas.height)

          // Draw heatmap only if we have attention data
          if (attentionData) {
            const gridH = analysisResult.attention_map!.image_features.unpadded.grid_h
            const gridW = analysisResult.attention_map!.image_features.unpadded.grid_w
            const cellW = unpaddedCanvas.width / gridW
            const cellH = unpaddedCanvas.height / gridH

            const weights = attentionData.to_image_unpadded

            // Get top-K patches with their indices and weights
            const weightedPatches = weights
              .map((weight, idx) => ({ weight, idx }))
              .filter(p => p.weight >= threshold[0])
              .sort((a, b) => b.weight - a.weight)
              .slice(0, showTopK)

            const maxWeight = weightedPatches.length > 0 ? weightedPatches[0].weight : 1

            weightedPatches.forEach(({ weight, idx }, rank) => {
              const row = Math.floor(idx / gridW)
              const col = idx % gridW
              const x = col * cellW
              const y = row * cellH

              const normalizedWeight = focusMode ? weight / maxWeight : weight
              const opacity = Math.min(0.3 + normalizedWeight * 0.7, 0.9)

              // Draw filled rectangle
              ctx.fillStyle = `rgba(59, 130, 246, ${opacity})`
              ctx.fillRect(x, y, cellW, cellH)

              // Draw border
              ctx.strokeStyle = `rgba(255, 255, 255, 0.8)`
              ctx.lineWidth = 1
              ctx.strokeRect(x, y, cellW, cellH)

              // Draw weight value
              ctx.fillStyle = 'white'
              ctx.font = 'bold 10px monospace'
              ctx.textAlign = 'center'
              ctx.textBaseline = 'middle'
              ctx.shadowColor = 'black'
              ctx.shadowBlur = 3
              const text = weight.toFixed(3)
              ctx.fillText(text, x + cellW / 2, y + cellH / 2)
              ctx.shadowBlur = 0
            })
          }
        }
      }
    }
  }, [attentionData, threshold, focusMode, selectedImage, analysisResult, showTopK])

  const outputTokens = analysisResult?.tokens.filter(t => t.io === 'output' && t.type === 'text') || []
  const inputTokens = analysisResult?.tokens.filter(t => t.io === 'input' && t.type === 'text') || []

  // Get text-to-text attention weights
  const textAttentionWeights = attentionData?.to_text || []

  if (!analysisResult || !selectedImage) {
    return (
      <div className="flex-1 flex items-center justify-center p-6">
        <div className="text-center text-gray-500 dark:text-gray-400">
          <p className="text-lg">Run analysis to view attention maps</p>
        </div>
      </div>
    )
  }

  return (
    <div className="flex-1 overflow-auto p-6">
      <div className="max-w-7xl mx-auto space-y-6">
        {/* Control Panel */}
        <div className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-4">
          <div className="space-y-4">
            <h3 className="text-lg font-medium text-gray-900 dark:text-white">
              Attention Controls
            </h3>
            <div className="flex items-center justify-between flex-wrap gap-4">
              <label className="flex items-center space-x-2 text-sm">
                <input
                  type="checkbox"
                  checked={focusMode}
                  onChange={(e) => setFocusMode(e.target.checked)}
                  className="rounded border-gray-300 dark:border-gray-600"
                />
                <span className="text-gray-700 dark:text-gray-300">Focus on Top-K</span>
              </label>

              <div className="flex items-center space-x-3">
                <span className="text-sm text-gray-700 dark:text-gray-300">Show Top:</span>
                <select
                  value={showTopK}
                  onChange={(e) => setShowTopK(Number(e.target.value))}
                  className="px-3 py-1 border border-gray-300 dark:border-gray-600 rounded text-sm bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
                >
                  <option value={5}>5</option>
                  <option value={10}>10</option>
                  <option value={20}>20</option>
                  <option value={30}>30</option>
                  <option value={50}>50</option>
                </select>
                <span className="text-sm text-gray-500 dark:text-gray-400">patches</span>
              </div>

              <div className="flex items-center space-x-3 min-w-[200px]">
                <span className="text-sm text-gray-700 dark:text-gray-300">Threshold:</span>
                <Slider
                  value={threshold}
                  onValueChange={setThreshold}
                  min={0}
                  max={1}
                  step={0.05}
                  className="flex-1 relative flex items-center select-none touch-none h-5"
                >
                  <div className="relative flex-1 h-1 bg-gray-200 dark:bg-gray-700 rounded-full">
                    <div
                      className="absolute h-full bg-blue-600 rounded-full"
                      style={{ width: `${threshold[0] * 100}%` }}
                    />
                  </div>
                  <div
                    className="block w-4 h-4 bg-blue-600 rounded-full shadow cursor-pointer"
                    style={{ marginLeft: `-0.5rem`, position: 'absolute', left: `${threshold[0] * 100}%` }}
                  />
                </Slider>
                <span className="text-sm font-medium text-gray-900 dark:text-white min-w-[3rem]">
                  {threshold[0].toFixed(2)}
                </span>
              </div>
            </div>
          </div>
        </div>

        {/* Image Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Padded Image */}
          <div className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-4">
            <h3 className="text-sm font-medium text-gray-900 dark:text-white mb-3">
              Padded Embedding ({analysisResult.attention_map?.image_features.padded.grid_h}×
              {analysisResult.attention_map?.image_features.padded.grid_w})
            </h3>
            <div className="relative bg-gray-100 dark:bg-gray-700 rounded-lg overflow-hidden">
              <canvas
                ref={paddedCanvasRef}
                width={400}
                height={400}
                className="w-full h-auto"
              />
            </div>
          </div>

          {/* Unpadded Image */}
          <div className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-4">
            <h3 className="text-sm font-medium text-gray-900 dark:text-white mb-3">
              Unpadded Embedding ({analysisResult.attention_map?.image_features.unpadded.grid_h}×
              {analysisResult.attention_map?.image_features.unpadded.grid_w})
            </h3>
            <div className="relative bg-gray-100 dark:bg-gray-700 rounded-lg overflow-hidden">
              <canvas
                ref={unpaddedCanvasRef}
                width={400}
                height={400}
                className="w-full h-auto"
              />
            </div>
          </div>
        </div>

        {/* Token Display */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Input Tokens */}
          <div className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-4">
            <h3 className="text-sm font-medium text-gray-900 dark:text-white mb-3">
              Input Tokens
            </h3>
            <div className="flex flex-wrap gap-2">
              {inputTokens.map((token, idx) => {
                const weight = textAttentionWeights[idx] || 0
                const isHighlighted = weight > 0.1
                return (
                  <span
                    key={token.position}
                    className={cn(
                      "px-2 py-1 rounded text-sm font-mono transition-colors cursor-default",
                      isHighlighted && activeTokenId
                        ? "bg-blue-100 dark:bg-blue-900 text-blue-900 dark:text-blue-100 font-semibold"
                        : "bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300"
                    )}
                    style={{
                      opacity: isHighlighted && activeTokenId ? 0.5 + (weight * 0.5) : 1
                    }}
                  >
                    {token.token}
                  </span>
                )
              })}
            </div>
          </div>

          {/* Output Tokens */}
          <div className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-4">
            <h3 className="text-sm font-medium text-gray-900 dark:text-white mb-3">
              Output Tokens (hover to see attention)
            </h3>
            <div className="flex flex-wrap gap-2">
              {outputTokens.map(token => (
                <span
                  key={token.position}
                  onMouseEnter={() => setHoveredTokenId(token.token_id)}
                  onMouseLeave={() => setHoveredTokenId(null)}
                  onClick={() => onTokenClick?.(token.token_id)}
                  className={cn(
                    "px-2 py-1 rounded text-sm font-mono transition-all cursor-pointer",
                    activeTokenId === token.token_id
                      ? "bg-blue-600 text-white shadow-lg font-bold"
                      : "bg-green-100 dark:bg-green-900 text-green-900 dark:text-green-100 hover:bg-green-200 dark:hover:bg-green-800"
                  )}
                >
                  {token.token}
                </span>
              ))}
            </div>
          </div>
        </div>

        {/* Instructions */}
        {!activeTokenId && (
          <div className="bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800 rounded-lg p-4">
            <p className="text-sm text-blue-900 dark:text-blue-100">
              💡 Hover over an output token to see which parts of the image and input text it&apos;s attending to
            </p>
          </div>
        )}
      </div>
    </div>
  )
}
