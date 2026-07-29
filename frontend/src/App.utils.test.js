import { describe, it, expect } from 'vitest'
import {
  confidenceToFraction,
  formatPercent,
  toLabel,
  normalizeLabel,
  isNeutralPrediction,
  hasHighConfidenceNeutral,
  splitKeywordText,
  uniqueList,
} from './App.jsx'

describe('confidenceToFraction', () => {
  it('passes through values already on a 0-1 scale', () => {
    expect(confidenceToFraction(0.42)).toBeCloseTo(0.42)
  })

  it('converts 0-100 scale values to a 0-1 fraction', () => {
    expect(confidenceToFraction(75)).toBeCloseTo(0.75)
  })

  it('returns 0 for non-numeric input', () => {
    expect(confidenceToFraction('not-a-number')).toBe(0)
  })

  it('clamps out-of-range values into [0, 1]', () => {
    expect(confidenceToFraction(250)).toBe(1)
    expect(confidenceToFraction(-5)).toBe(0)
  })
})

describe('formatPercent', () => {
  it('formats a 0-1 fraction as a percentage string', () => {
    expect(formatPercent(0.4567)).toBe('45.67%')
  })
})

describe('toLabel / normalizeLabel', () => {
  it('capitalizes each word\'s first letter and strips separators, without altering other letters', () => {
    // toLabel only uppercases word-initial characters; it does not lowercase
    // the rest, so an acronym like "BA" is preserved as-is.
    expect(toLabel('BA-cellulitis')).toBe('BA Cellulitis')
    expect(toLabel('vi_chickenpox')).toBe('Vi Chickenpox')
  })

  it('normalizes labels to a comparable lowercase form', () => {
    expect(normalizeLabel('Normal Skin')).toBe('normal skin')
    expect(normalizeLabel('normal_skin')).toBe('normal skin')
  })
})

describe('isNeutralPrediction / hasHighConfidenceNeutral', () => {
  it('recognizes known neutral labels regardless of case/separators', () => {
    expect(isNeutralPrediction('Normal')).toBe(true)
    expect(isNeutralPrediction('Skin Not Found')).toBe(true)
    expect(isNeutralPrediction('Scabies')).toBe(false)
  })

  it('flags rows only when a neutral label meets the confidence threshold', () => {
    const rows = [{ prediction: 'Normal', confidence: 0.85 }]
    expect(hasHighConfidenceNeutral(rows)).toBe(true)
  })

  it('does not flag a neutral label below the threshold', () => {
    const rows = [{ prediction: 'Normal', confidence: 0.5 }]
    expect(hasHighConfidenceNeutral(rows)).toBe(false)
  })

  it('does not flag a non-neutral high-confidence prediction', () => {
    const rows = [{ prediction: 'Scabies', confidence: 0.95 }]
    expect(hasHighConfidenceNeutral(rows)).toBe(false)
  })
})

describe('splitKeywordText / uniqueList', () => {
  it('splits on commas, semicolons, slashes, and newlines', () => {
    expect(splitKeywordText('itching, redness; scaling/peeling\nfever')).toEqual([
      'itching',
      'redness',
      'scaling',
      'peeling',
      'fever',
    ])
  })

  it('drops empty segments', () => {
    expect(splitKeywordText('itching,,  ,redness')).toEqual(['itching', 'redness'])
  })

  it('deduplicates while preserving first-seen order', () => {
    expect(uniqueList(['a', 'b', 'a', 'c', 'b'])).toEqual(['a', 'b', 'c'])
  })
})
