"use client"

import { Play, Eye, BarChart3, RotateCcw } from "lucide-react"
import { cn } from "@/lib/utils"
import { ViewType } from "@/lib/types"
import { modelConfigurations } from "@/lib/mockData"

interface ToolbarProps {
  selectedModel: string
  onModelChange: (modelId: string) => void
  onRunAnalysis: () => void
  onReset?: () => void
  isRunning: boolean
  hasImage: boolean
  currentView: ViewType
  onViewChange: (view: ViewType) => void
  hasResults: boolean
}

export function Toolbar({
  selectedModel,
  onModelChange,
  onRunAnalysis,
  onReset,
  isRunning,
  hasImage,
  currentView,
  onViewChange,
  hasResults
}: ToolbarProps) {
  return (
    <div className="bg-white dark:bg-gray-800 border-b border-gray-200 dark:border-gray-700 px-6 py-3">
      <div className="flex items-center justify-between">
        {/* Left: Model Selector */}
        <div className="flex items-center space-x-4">
          <label className="text-sm font-medium text-gray-700 dark:text-gray-300">
            Model:
          </label>
          <select
            value={selectedModel}
            onChange={(e) => onModelChange(e.target.value)}
            className="px-3 py-1.5 border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-700 text-sm text-gray-900 dark:text-white focus:ring-2 focus:ring-blue-500 focus:border-transparent min-w-[280px]"
          >
            {modelConfigurations.map((model) => (
              <option key={model.id} value={model.id}>
                {model.name}
              </option>
            ))}
          </select>
        </div>

        {/* Center: Run Analysis and Reset Buttons */}
        <div className="flex items-center space-x-3">
          <button
            onClick={onRunAnalysis}
            disabled={isRunning || !hasImage}
            className={cn(
              "flex items-center space-x-2 px-6 py-2 rounded-md font-medium transition-colors",
              isRunning || !hasImage
                ? "bg-blue-400 dark:bg-blue-600 text-white cursor-not-allowed"
                : "bg-blue-600 hover:bg-blue-700 text-white"
            )}
          >
            <Play className="h-4 w-4" />
            <span>{isRunning ? "Running..." : "Run Analysis"}</span>
          </button>

          {onReset && (
            <button
              onClick={onReset}
              disabled={!hasResults}
              className={cn(
                "flex items-center space-x-2 px-4 py-2 rounded-md font-medium transition-colors",
                !hasResults
                  ? "bg-gray-200 dark:bg-gray-700 text-gray-400 dark:text-gray-500 cursor-not-allowed"
                  : "bg-gray-200 dark:bg-gray-700 hover:bg-gray-300 dark:hover:bg-gray-600 text-gray-700 dark:text-gray-300"
              )}
            >
              <RotateCcw className="h-4 w-4" />
              <span>Reset</span>
            </button>
          )}
        </div>

        {/* Right: View Tabs */}
        <div className="flex items-center space-x-2 bg-gray-100 dark:bg-gray-700 rounded-lg p-1">
          <button
            onClick={() => onViewChange('overview')}
            className={cn(
              "flex items-center space-x-2 px-4 py-1.5 rounded-md text-sm font-medium transition-colors",
              currentView === 'overview'
                ? "bg-white dark:bg-gray-600 text-blue-600 dark:text-blue-400 shadow-sm"
                : "text-gray-600 dark:text-gray-300 hover:text-gray-900 dark:hover:text-white"
            )}
          >
            <BarChart3 className="h-4 w-4" />
            <span>Overview</span>
          </button>
          <button
            onClick={() => hasResults && onViewChange('attention')}
            disabled={!hasResults}
            className={cn(
              "flex items-center space-x-2 px-4 py-1.5 rounded-md text-sm font-medium transition-colors",
              currentView === 'attention'
                ? "bg-white dark:bg-gray-600 text-blue-600 dark:text-blue-400 shadow-sm"
                : !hasResults
                ? "text-gray-400 dark:text-gray-500 cursor-not-allowed"
                : "text-gray-600 dark:text-gray-300 hover:text-gray-900 dark:hover:text-white"
            )}
          >
            <Eye className="h-4 w-4" />
            <span>Attention Map</span>
          </button>
          <button
            onClick={() => hasResults && onViewChange('prediction')}
            disabled={!hasResults}
            className={cn(
              "flex items-center space-x-2 px-4 py-1.5 rounded-md text-sm font-medium transition-colors",
              currentView === 'prediction'
                ? "bg-white dark:bg-gray-600 text-blue-600 dark:text-blue-400 shadow-sm"
                : !hasResults
                ? "text-gray-400 dark:text-gray-500 cursor-not-allowed"
                : "text-gray-600 dark:text-gray-300 hover:text-gray-900 dark:hover:text-white"
            )}
          >
            <BarChart3 className="h-4 w-4" />
            <span>Token Prediction</span>
          </button>
        </div>
      </div>
    </div>
  )
}
