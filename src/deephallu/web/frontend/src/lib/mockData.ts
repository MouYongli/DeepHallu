import {
  ImageMetadata,
  AnalysisResult,
  Token,
  AttentionMapData,
  PredictionData,
  ModelConfig
} from './types'

// Import JSON data
import imagesData from '../data/images.json'
import modelsData from '../data/models.json'
import analysisTemplate from '../data/analysis-template.json'
import scoresData from '../data/scores.json'
import sequencesData from '../data/sequences.json'

// Mock images with questions/answers
export const mockImages: ImageMetadata[] = imagesData as ImageMetadata[]

export const modelConfigurations: ModelConfig[] = modelsData as ModelConfig[]

// Use real token sequences from JSON
const getRealTokens = (): Token[] => {
  // Type cast the imported sequences data to Token array
  return sequencesData as Token[]
}

// Generate mock attention map data
const generateMockAttentionMap = (tokens: Token[]): AttentionMapData => {
  const outputTokens = tokens.filter(t => t.io === 'output' && t.type === 'text')
  const config = analysisTemplate.token_config

  return {
    image_features: {
      padded: {
        grid_h: config.grid_size.padded.height,
        grid_w: config.grid_size.padded.width
      },
      unpadded: {
        grid_h: config.grid_size.unpadded.height,
        grid_w: config.grid_size.unpadded.width,
        valid_mask: Array(config.grid_size.unpadded.height * config.grid_size.unpadded.width).fill(0).map((_, i) =>
          (i % config.grid_size.padded.width < config.grid_size.unpadded.width &&
           Math.floor(i / config.grid_size.padded.width) < config.grid_size.unpadded.height) ? 1 : 0
        )
      }
    },
    text_tokens: tokens
      .filter(t => t.type === 'text')
      .map(t => ({
        id: t.token_id,
        token: t.token,
        io: t.io,
        pos: t.position
      })),
    attn: outputTokens.map(token => {
      // Generate random attention weights
      const paddedSize = config.grid_size.padded.height * config.grid_size.padded.width
      const unpaddedSize = config.grid_size.unpadded.height * config.grid_size.unpadded.width
      const toPaddedImage = Array(paddedSize).fill(0).map(() => Math.random())
      const toUnpaddedImage = Array(unpaddedSize).fill(0).map(() => Math.random())
      const toText = Array(tokens.filter(t => t.type === 'text' && t.io === 'input').length)
        .fill(0).map(() => Math.random())

      // Normalize
      const sumPadded = toPaddedImage.reduce((a, b) => a + b, 0)
      const sumUnpadded = toUnpaddedImage.reduce((a, b) => a + b, 0)
      const sumText = toText.reduce((a, b) => a + b, 0)

      return {
        token_id: token.token_id,
        to_image_padded: toPaddedImage.map(v => v / sumPadded),
        to_image_unpadded: toUnpaddedImage.map(v => v / sumUnpadded),
        to_text: toText.map(v => v / sumText)
      }
    })
  }
}

// Use real prediction scores from JSON
const getRealPredictions = (tokens: Token[]): PredictionData => {
  // Type the scores data
  type ScoreEntry = {
    step: number
    entropy_nats: number
    entropy_bits: number
    top_k_tokens: string[]
    top_k_probs: number[]
    top_k_token_ids: number[]
  }

  const scores = scoresData as ScoreEntry[]
  const outputTokens = tokens.filter(t => t.io === 'output' && t.type === 'text')

  return {
    predictions: outputTokens.map((token, idx) => {
      // Use the corresponding score entry if available, otherwise use the last one
      const scoreEntry = scores[Math.min(idx, scores.length - 1)]

      return {
        context_pos: token.position - 1,
        topk: scoreEntry.top_k_tokens.map((t, i) => ({
          token: t,
          prob: scoreEntry.top_k_probs[i]
        })),
        entropy: scoreEntry.entropy_nats,
        actual_token: token.token
      }
    })
  }
}

// Mock analysis function
export const runMockAnalysis = async (
  imageId: string,
  modelId: string,
  prompt: string,
  answer: string = ""
): Promise<AnalysisResult> => {
  // Simulate API delay
  const minTime = analysisTemplate.inference_config.min_inference_time
  const maxTime = analysisTemplate.inference_config.max_inference_time
  await new Promise(resolve => setTimeout(resolve, minTime + Math.random() * (maxTime - minTime)))

  const generatedText = analysisTemplate.generated_text

  const tokens = getRealTokens()
  const attentionMap = generateMockAttentionMap(tokens)
  const predictions = getRealPredictions(tokens)

  const imageTokenCount = tokens.filter(t => t.type === 'image').length
  const textTokenCount = tokens.filter(t => t.type === 'text').length
  const inputTokenCount = tokens.filter(t => t.io === 'input').length
  const outputTokenCount = tokens.filter(t => t.io === 'output').length

  return {
    id: `analysis_${Date.now()}`,
    image_id: imageId,
    model: modelId,
    prompt,
    answer,
    generated_text: generatedText,
    total_tokens: tokens.length,
    image_tokens: imageTokenCount,
    text_tokens: textTokenCount,
    input_tokens: inputTokenCount,
    output_tokens: outputTokenCount,
    inference_time: minTime + Math.random() * (maxTime - minTime),
    tokens,
    attention_map: attentionMap,
    predictions,
    timestamp: new Date()
  }
}
