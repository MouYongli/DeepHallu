// Image metadata
export interface ImageMetadata {
  id: string
  name: string
  path: string
  category: string
  dataset: string
  resolution: string
  format: string
  size: string
  questions?: QuestionAnswer[]
}

// Question and Answer pairs
export interface QuestionAnswer {
  question: string
  answer: string
}

// Token data structure
export interface Token {
  token_id: number
  token: string
  type: 'image' | 'text'
  io: 'input' | 'output'
  position: number
}

// Attention Map data structure
export interface AttentionMapData {
  image_features: {
    padded: {
      grid_h: number
      grid_w: number
    }
    unpadded: {
      grid_h: number
      grid_w: number
      valid_mask: number[]
    }
  }
  text_tokens: Array<{
    id: number
    token: string
    io: 'input' | 'output'
    pos: number
  }>
  attn: Array<{
    token_id: number
    to_image_padded: number[]
    to_image_unpadded: number[]
    to_text: number[]
  }>
}

// Token Prediction data structure
export interface TokenPrediction {
  context_pos: number
  topk: Array<{
    token: string
    prob: number
  }>
  entropy: number
  actual_token?: string
}

export interface PredictionData {
  predictions: TokenPrediction[]
}

// Analysis result
export interface AnalysisResult {
  id: string
  image_id: string
  model: string
  prompt: string
  answer: string
  generated_text: string

  // Statistics
  total_tokens: number
  image_tokens: number
  text_tokens: number
  input_tokens: number
  output_tokens: number
  inference_time: number // in ms

  // Token list
  tokens: Token[]

  // Attention data
  attention_map?: AttentionMapData

  // Prediction data
  predictions?: PredictionData

  timestamp: Date
}

// Model configuration
export interface ModelConfig {
  id: string
  name: string
  parameters: string
  memoryUsage: string
  inferenceTime: string
  accuracy: string
  hallucinationRate: string
}

// View types
export type ViewType = 'overview' | 'attention' | 'prediction'
