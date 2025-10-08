"use client"

import { useState } from "react"
import { Image as ImageIcon, Clock, Filter } from "lucide-react"
import { cn } from "@/lib/utils"
import { ImageMetadata, AnalysisResult } from "@/lib/types"

interface OverviewViewProps {
  selectedImage: ImageMetadata | null
  analysisResult: AnalysisResult | null
  prompt: string
  setPrompt: (prompt: string) => void
  answer: string
  setAnswer: (answer: string) => void
  onTokenClick?: (tokenId: number) => void
  selectedTokenId?: number
}

export function OverviewView({
  selectedImage,
  analysisResult,
  prompt,
  setPrompt,
  answer,
  setAnswer,
  onTokenClick,
  selectedTokenId
}: OverviewViewProps) {
  const [tokenFilter, setTokenFilter] = useState<{
    type: 'all' | 'image' | 'text'
    io: 'all' | 'input' | 'output'
  }>({ type: 'all', io: 'all' })

  const filteredTokens = analysisResult?.tokens.filter(token => {
    const typeMatch = tokenFilter.type === 'all' || token.type === tokenFilter.type
    const ioMatch = tokenFilter.io === 'all' || token.io === tokenFilter.io
    return typeMatch && ioMatch
  }) || []

  return (
    <div className="flex-1 overflow-auto p-6">
      <div className="max-w-7xl mx-auto space-y-6">
        {/* Top Row: Image and Prompt Panel */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Image Display */}
          <div className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-4">
            <div className="flex items-center space-x-2 mb-4">
              <ImageIcon className="h-5 w-5 text-gray-600 dark:text-gray-400" />
              <h3 className="text-lg font-medium text-gray-900 dark:text-white">
                Input Image
              </h3>
            </div>

            <div className="w-full bg-gray-100 dark:bg-gray-700 rounded-lg border border-gray-300 dark:border-gray-600 flex items-center justify-center min-h-[300px]">
              {selectedImage ? (
                <div className="text-center w-full p-4">
                  <img
                    src={selectedImage.path}
                    alt={selectedImage.name}
                    className="max-w-full max-h-[280px] object-contain rounded-lg mx-auto"
                  />
                  <p className="text-sm text-gray-700 dark:text-gray-300 font-medium mt-2">
                    {selectedImage.name}
                  </p>
                </div>
              ) : (
                <div className="text-center">
                  <ImageIcon className="h-12 w-12 text-gray-400 mx-auto mb-2" />
                  <p className="text-gray-500 dark:text-gray-400">
                    No image selected
                  </p>
                </div>
              )}
            </div>

            {selectedImage && (
              <div className="mt-4 grid grid-cols-2 gap-3 text-sm">
                <div>
                  <span className="text-gray-500 dark:text-gray-400">Dataset:</span>
                  <span className="ml-2 text-gray-900 dark:text-white">{selectedImage.dataset}</span>
                </div>
                <div>
                  <span className="text-gray-500 dark:text-gray-400">Category:</span>
                  <span className="ml-2 text-gray-900 dark:text-white">{selectedImage.category}</span>
                </div>
                <div>
                  <span className="text-gray-500 dark:text-gray-400">Resolution:</span>
                  <span className="ml-2 text-gray-900 dark:text-white">{selectedImage.resolution}</span>
                </div>
                <div>
                  <span className="text-gray-500 dark:text-gray-400">Format:</span>
                  <span className="ml-2 text-gray-900 dark:text-white">{selectedImage.format}</span>
                </div>
              </div>
            )}
          </div>

          {/* Prompt Panel */}
          <div className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-4">
            <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-4">
              Prompt Configuration
            </h3>

            <div className="space-y-4">
              {/* Question */}
              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                  Question
                </label>
                {selectedImage?.questions && selectedImage.questions.length > 0 ? (
                  <select
                    value={prompt}
                    onChange={(e) => {
                      const newPrompt = e.target.value
                      setPrompt(newPrompt)

                      // Find the corresponding answer for the selected question
                      const selectedQuestion = selectedImage.questions?.find(qa => qa.question === newPrompt)
                      if (selectedQuestion) {
                        setAnswer(selectedQuestion.answer)
                      } else {
                        // If custom prompt is selected, clear the answer
                        setAnswer("")
                      }
                    }}
                    className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-700 text-sm text-gray-900 dark:text-white focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  >
                    {selectedImage.questions.map((qa, idx) => (
                      <option key={idx} value={qa.question}>
                        {qa.question}
                      </option>
                    ))}
                    <option value="">Custom prompt...</option>
                  </select>
                ) : (
                  <input
                    type="text"
                    value={prompt}
                    onChange={(e) => setPrompt(e.target.value)}
                    placeholder="Enter your question..."
                    className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-700 text-sm text-gray-900 dark:text-white focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  />
                )}
              </div>

              {/* Custom Prompt if needed */}
              {selectedImage?.questions?.some(qa => qa.question === prompt) === false && (
                <div>
                  <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                    Custom Prompt
                  </label>
                  <textarea
                    value={prompt}
                    onChange={(e) => setPrompt(e.target.value)}
                    placeholder="Enter custom prompt..."
                    rows={3}
                    className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-700 text-sm text-gray-900 dark:text-white focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  />
                </div>
              )}

              {/* Answer */}
              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                  Expected Answer (Optional)
                </label>
                <textarea
                  value={answer}
                  onChange={(e) => setAnswer(e.target.value)}
                  placeholder="Enter expected answer for comparison..."
                  rows={3}
                  className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-700 text-sm text-gray-900 dark:text-white focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                />
              </div>
            </div>
          </div>
        </div>

        {/* Results Section (only show if analysis has been run) */}
        {analysisResult && (
          <>
            {/* Generated Text */}
            <div className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-4">
              <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-3">
                Model Response
              </h3>
              <div className="bg-gray-50 dark:bg-gray-700 rounded-lg p-4">
                <p className="text-gray-900 dark:text-white text-sm leading-relaxed">
                  {analysisResult.generated_text}
                </p>
              </div>
            </div>

            {/* Statistics Cards */}
            <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-4">
              <div className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-4">
                <div className="text-xs text-gray-500 dark:text-gray-400 mb-1">Total Tokens</div>
                <div className="text-2xl font-bold text-gray-900 dark:text-white">
                  {analysisResult.total_tokens}
                </div>
              </div>

              <div className="bg-blue-50 dark:bg-blue-900/20 rounded-lg border border-blue-200 dark:border-blue-800 p-4">
                <div className="text-xs text-blue-700 dark:text-blue-300 mb-1">Image Tokens</div>
                <div className="text-2xl font-bold text-blue-900 dark:text-blue-100">
                  {analysisResult.image_tokens}
                </div>
              </div>

              <div className="bg-green-50 dark:bg-green-900/20 rounded-lg border border-green-200 dark:border-green-800 p-4">
                <div className="text-xs text-green-700 dark:text-green-300 mb-1">Text Tokens</div>
                <div className="text-2xl font-bold text-green-900 dark:text-green-100">
                  {analysisResult.text_tokens}
                </div>
              </div>

              <div className="bg-purple-50 dark:bg-purple-900/20 rounded-lg border border-purple-200 dark:border-purple-800 p-4">
                <div className="text-xs text-purple-700 dark:text-purple-300 mb-1">Input Tokens</div>
                <div className="text-2xl font-bold text-purple-900 dark:text-purple-100">
                  {analysisResult.input_tokens}
                </div>
              </div>

              <div className="bg-orange-50 dark:bg-orange-900/20 rounded-lg border border-orange-200 dark:border-orange-800 p-4">
                <div className="text-xs text-orange-700 dark:text-orange-300 mb-1">Output Tokens</div>
                <div className="text-2xl font-bold text-orange-900 dark:text-orange-100">
                  {analysisResult.output_tokens}
                </div>
              </div>

              <div className="bg-gray-100 dark:bg-gray-700 rounded-lg border border-gray-300 dark:border-gray-600 p-4">
                <div className="text-xs text-gray-600 dark:text-gray-400 mb-1 flex items-center">
                  <Clock className="h-3 w-3 mr-1" />
                  Inference Time
                </div>
                <div className="text-2xl font-bold text-gray-900 dark:text-white">
                  {(analysisResult.inference_time / 1000).toFixed(2)}s
                </div>
              </div>
            </div>

            {/* Token Table */}
            <div className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-4">
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-lg font-medium text-gray-900 dark:text-white">
                  Token List
                </h3>
                <div className="flex items-center space-x-2">
                  <Filter className="h-4 w-4 text-gray-500" />
                  <select
                    value={tokenFilter.type}
                    onChange={(e) => setTokenFilter(prev => ({ ...prev, type: e.target.value as 'all' | 'image' | 'text' }))}
                    className="px-2 py-1 border border-gray-300 dark:border-gray-600 rounded text-xs bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
                  >
                    <option value="all">All Types</option>
                    <option value="image">Image Only</option>
                    <option value="text">Text Only</option>
                  </select>
                  <select
                    value={tokenFilter.io}
                    onChange={(e) => setTokenFilter(prev => ({ ...prev, io: e.target.value as 'all' | 'input' | 'output' }))}
                    className="px-2 py-1 border border-gray-300 dark:border-gray-600 rounded text-xs bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
                  >
                    <option value="all">All I/O</option>
                    <option value="input">Input Only</option>
                    <option value="output">Output Only</option>
                  </select>
                </div>
              </div>

              <div className="overflow-auto max-h-96 border border-gray-200 dark:border-gray-700 rounded-lg">
                <table className="w-full text-sm">
                  <thead className="bg-gray-50 dark:bg-gray-700 sticky top-0">
                    <tr>
                      <th className="px-4 py-2 text-left text-xs font-medium text-gray-700 dark:text-gray-300">ID</th>
                      <th className="px-4 py-2 text-left text-xs font-medium text-gray-700 dark:text-gray-300">Token</th>
                      <th className="px-4 py-2 text-left text-xs font-medium text-gray-700 dark:text-gray-300">Type</th>
                      <th className="px-4 py-2 text-left text-xs font-medium text-gray-700 dark:text-gray-300">I/O</th>
                      <th className="px-4 py-2 text-left text-xs font-medium text-gray-700 dark:text-gray-300">Position</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-gray-200 dark:divide-gray-700">
                    {filteredTokens.slice(0, 100).map((token) => (
                      <tr
                        key={token.position}
                        onClick={() => onTokenClick?.(token.token_id)}
                        className={cn(
                          "cursor-pointer transition-colors",
                          selectedTokenId === token.token_id
                            ? "bg-blue-100 dark:bg-blue-900/50"
                            : "hover:bg-gray-50 dark:hover:bg-gray-700"
                        )}
                      >
                        <td className="px-4 py-2 text-gray-900 dark:text-white">{token.token_id}</td>
                        <td className="px-4 py-2 font-mono text-gray-900 dark:text-white">{token.token}</td>
                        <td className="px-4 py-2">
                          <span className={cn(
                            "px-2 py-1 rounded text-xs font-medium",
                            token.type === 'image'
                              ? "bg-blue-100 text-blue-700 dark:bg-blue-900 dark:text-blue-300"
                              : "bg-green-100 text-green-700 dark:bg-green-900 dark:text-green-300"
                          )}>
                            {token.type}
                          </span>
                        </td>
                        <td className="px-4 py-2">
                          <span className={cn(
                            "px-2 py-1 rounded text-xs font-medium",
                            token.io === 'input'
                              ? "bg-purple-100 text-purple-700 dark:bg-purple-900 dark:text-purple-300"
                              : "bg-orange-100 text-orange-700 dark:bg-orange-900 dark:text-orange-300"
                          )}>
                            {token.io}
                          </span>
                        </td>
                        <td className="px-4 py-2 text-gray-900 dark:text-white">{token.position}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
                {filteredTokens.length > 100 && (
                  <div className="px-4 py-3 bg-gray-50 dark:bg-gray-700 text-center text-xs text-gray-500 dark:text-gray-400">
                    Showing first 100 of {filteredTokens.length} tokens
                  </div>
                )}
              </div>
            </div>
          </>
        )}
      </div>
    </div>
  )
}
