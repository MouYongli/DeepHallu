"use client"

import { useState } from "react"
import { AnalysisResult, TokenPrediction } from "@/lib/types"
import { cn } from "@/lib/utils"
import { TrendingUp, Info } from "lucide-react"

interface TokenPredictionViewProps {
  analysisResult: AnalysisResult | null
  selectedTokenId?: number
  onTokenClick?: (tokenId: number) => void
}

export function TokenPredictionView({
  analysisResult,
  selectedTokenId,
  onTokenClick
}: TokenPredictionViewProps) {
  const [hoveredTokenId, setHoveredTokenId] = useState<number | null>(null)
  const [entropyUnit, setEntropyUnit] = useState<'nat' | 'bit'>('nat')

  const activeTokenId = hoveredTokenId ?? selectedTokenId

  if (!analysisResult) {
    return (
      <div className="flex-1 flex items-center justify-center p-6">
        <div className="text-center text-gray-500 dark:text-gray-400">
          <p className="text-lg">Run analysis to view token predictions</p>
        </div>
      </div>
    )
  }

  const inputTokens = analysisResult.tokens.filter(t => t.io === 'input' && t.type === 'text')
  const outputTokens = analysisResult.tokens.filter(t => t.io === 'output' && t.type === 'text')

  // Find prediction for the active token
  // Prediction shows what was predicted BEFORE this token was generated
  const activePrediction = analysisResult.predictions?.predictions.find(
    p => p.context_pos + 1 === analysisResult.tokens.find(t => t.token_id === activeTokenId)?.position
  )

  // For the last input token, show the first prediction
  const lastInputToken = inputTokens[inputTokens.length - 1]
  const firstPrediction = analysisResult.predictions?.predictions[0]

  const displayPrediction = activeTokenId === lastInputToken?.token_id
    ? firstPrediction
    : activePrediction

  const convertEntropy = (entropy: number) => {
    return entropyUnit === 'bit' ? entropy / Math.LN2 : entropy
  }

  return (
    <div className="flex-1 overflow-auto p-6">
      <div className="max-w-7xl mx-auto space-y-6">
        {/* Control Panel */}
        <div className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-4">
          <div className="flex items-center justify-between">
            <h3 className="text-lg font-medium text-gray-900 dark:text-white">
              Token Prediction Analysis
            </h3>
            <div className="flex items-center space-x-2">
              <span className="text-sm text-gray-700 dark:text-gray-300">Entropy Unit:</span>
              <button
                onClick={() => setEntropyUnit('nat')}
                className={cn(
                  "px-3 py-1 rounded-l-md text-sm font-medium transition-colors",
                  entropyUnit === 'nat'
                    ? "bg-blue-600 text-white"
                    : "bg-gray-200 dark:bg-gray-700 text-gray-700 dark:text-gray-300"
                )}
              >
                nat
              </button>
              <button
                onClick={() => setEntropyUnit('bit')}
                className={cn(
                  "px-3 py-1 rounded-r-md text-sm font-medium transition-colors",
                  entropyUnit === 'bit'
                    ? "bg-blue-600 text-white"
                    : "bg-gray-200 dark:bg-gray-700 text-gray-700 dark:text-gray-300"
                )}
              >
                bit
              </button>
            </div>
          </div>
        </div>

        {/* Token Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Input Tokens */}
          <div className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-4">
            <h3 className="text-sm font-medium text-gray-900 dark:text-white mb-3">
              Input Tokens
            </h3>
            <div className="flex flex-wrap gap-2">
              {inputTokens.map((token, idx) => {
                const isLast = idx === inputTokens.length - 1
                return (
                  <span
                    key={token.position}
                    onMouseEnter={() => isLast && setHoveredTokenId(token.token_id)}
                    onMouseLeave={() => setHoveredTokenId(null)}
                    onClick={() => isLast && onTokenClick?.(token.token_id)}
                    className={cn(
                      "px-2 py-1 rounded text-sm font-mono transition-all",
                      isLast
                        ? activeTokenId === token.token_id
                          ? "bg-purple-600 text-white shadow-lg scale-105 cursor-pointer"
                          : "bg-purple-100 dark:bg-purple-900 text-purple-900 dark:text-purple-100 hover:bg-purple-200 dark:hover:bg-purple-800 cursor-pointer"
                        : "bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300"
                    )}
                  >
                    {token.token}
                  </span>
                )
              })}
            </div>
            <p className="text-xs text-gray-500 dark:text-gray-400 mt-2">
              Hover over the last input token to see first prediction
            </p>
          </div>

          {/* Output Tokens */}
          <div className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-4">
            <h3 className="text-sm font-medium text-gray-900 dark:text-white mb-3">
              Output Tokens (hover to see prediction)
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
                      ? "bg-orange-600 text-white shadow-lg scale-105"
                      : "bg-orange-100 dark:bg-orange-900 text-orange-900 dark:text-orange-100 hover:bg-orange-200 dark:hover:bg-orange-800"
                  )}
                >
                  {token.token}
                </span>
              ))}
            </div>
          </div>
        </div>

        {/* Prediction Display */}
        {displayPrediction ? (
          <div className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-6">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-lg font-medium text-gray-900 dark:text-white flex items-center">
                <TrendingUp className="h-5 w-5 mr-2" />
                Next Token Prediction Distribution
              </h3>
              <div className="flex items-center space-x-4">
                <div className="text-sm">
                  <span className="text-gray-500 dark:text-gray-400">Context Position: </span>
                  <span className="font-mono font-medium text-gray-900 dark:text-white">
                    {displayPrediction.context_pos}
                  </span>
                </div>
                <div className="text-sm">
                  <span className="text-gray-500 dark:text-gray-400">Entropy: </span>
                  <span className="font-mono font-medium text-gray-900 dark:text-white">
                    {convertEntropy(displayPrediction.entropy).toFixed(3)} {entropyUnit}
                  </span>
                </div>
              </div>
            </div>

            {/* Top-K Predictions */}
            <div className="space-y-3">
              {displayPrediction.topk.map((pred, idx) => {
                const isActual = pred.token === displayPrediction.actual_token
                return (
                  <div key={idx} className="space-y-1">
                    <div className="flex items-center justify-between text-sm">
                      <div className="flex items-center space-x-2">
                        <span className="text-gray-500 dark:text-gray-400 w-6">#{idx + 1}</span>
                        <span className={cn(
                          "font-mono px-2 py-0.5 rounded",
                          isActual
                            ? "bg-green-100 dark:bg-green-900 text-green-900 dark:text-green-100 font-semibold"
                            : "bg-gray-100 dark:bg-gray-700 text-gray-900 dark:text-white"
                        )}>
                          {pred.token}
                        </span>
                        {isActual && (
                          <span className="text-xs text-green-600 dark:text-green-400 font-medium">
                            (actual)
                          </span>
                        )}
                      </div>
                      <span className="font-mono font-medium text-gray-900 dark:text-white">
                        {(pred.prob * 100).toFixed(2)}%
                      </span>
                    </div>
                    <div className="relative h-2 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden">
                      <div
                        className={cn(
                          "absolute left-0 top-0 h-full rounded-full transition-all",
                          isActual
                            ? "bg-green-600 dark:bg-green-500"
                            : "bg-blue-600 dark:bg-blue-500"
                        )}
                        style={{ width: `${pred.prob * 100}%` }}
                      />
                    </div>
                  </div>
                )
              })}
            </div>

            {/* Entropy Explanation */}
            <div className="mt-6 bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800 rounded-lg p-4">
              <div className="flex items-start space-x-2">
                <Info className="h-4 w-4 text-blue-600 dark:text-blue-400 mt-0.5 flex-shrink-0" />
                <div className="text-xs text-blue-900 dark:text-blue-100">
                  <p className="font-medium mb-1">Entropy Interpretation:</p>
                  <p>
                    Entropy measures the uncertainty of the prediction distribution.
                    Lower entropy ({convertEntropy(displayPrediction.entropy) < 1 ? 'like this one' : 'e.g., &lt; 1.0'})
                    indicates the model is confident (distribution is peaked),
                    while higher entropy indicates uncertainty (distribution is flat).
                  </p>
                </div>
              </div>
            </div>
          </div>
        ) : (
          <div className="bg-gray-50 dark:bg-gray-800 border-2 border-dashed border-gray-300 dark:border-gray-600 rounded-lg p-12">
            <div className="text-center text-gray-500 dark:text-gray-400">
              <TrendingUp className="h-12 w-12 mx-auto mb-4 opacity-50" />
              <p className="text-lg">Hover over a token to see its prediction distribution</p>
              <p className="text-sm mt-2">
                For output tokens, you&apos;ll see what was predicted before that token was generated
              </p>
            </div>
          </div>
        )}

        {/* Statistics Summary */}
        {analysisResult.predictions && (
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-4">
              <div className="text-xs text-gray-500 dark:text-gray-400 mb-1">Total Predictions</div>
              <div className="text-2xl font-bold text-gray-900 dark:text-white">
                {analysisResult.predictions.predictions.length}
              </div>
            </div>

            <div className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-4">
              <div className="text-xs text-gray-500 dark:text-gray-400 mb-1">Avg Entropy ({entropyUnit})</div>
              <div className="text-2xl font-bold text-gray-900 dark:text-white">
                {(analysisResult.predictions.predictions.reduce((acc, p) => acc + convertEntropy(p.entropy), 0) /
                  analysisResult.predictions.predictions.length).toFixed(3)}
              </div>
            </div>

            <div className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-4">
              <div className="text-xs text-gray-500 dark:text-gray-400 mb-1">Max Entropy ({entropyUnit})</div>
              <div className="text-2xl font-bold text-gray-900 dark:text-white">
                {Math.max(...analysisResult.predictions.predictions.map(p => convertEntropy(p.entropy))).toFixed(3)}
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
