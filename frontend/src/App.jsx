import { useEffect, useMemo, useState } from 'react'
import './App.css'
import logoImage from './assets/skinova-logo.png'

const API_BASE = import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000'
const NEUTRAL_LABELS = new Set(['normal', 'normal skin', 'normalskin', 'healthy', 'none', 'skin not found'])
const NEUTRAL_SUPPRESS_THRESHOLD = 0.8

export function confidenceToFraction(value) {
  const numeric = Number(value)
  if (!Number.isFinite(numeric)) {
    return 0
  }
  if (numeric > 1) {
    return Math.max(0, Math.min(1, numeric / 100))
  }
  return Math.max(0, Math.min(1, numeric))
}

export function formatPercent(value) {
  return `${(confidenceToFraction(value) * 100).toFixed(2)}%`
}

export function toLabel(value) {
  return String(value || '')
    .replace(/[_-]+/g, ' ')
    .replace(/\s+/g, ' ')
    .trim()
    .replace(/\b\w/g, (char) => char.toUpperCase())
}

export function normalizeLabel(value) {
  return String(value || '')
    .toLowerCase()
    .replace(/[_-]+/g, ' ')
    .replace(/\s+/g, ' ')
    .trim()
}

export function isNeutralPrediction(value) {
  return NEUTRAL_LABELS.has(normalizeLabel(value))
}

export function hasHighConfidenceNeutral(rows, threshold = NEUTRAL_SUPPRESS_THRESHOLD) {
  return rows.some(
    (row) => isNeutralPrediction(row.prediction) && confidenceToFraction(row.confidence) >= threshold
  )
}

export function splitKeywordText(raw) {
  return String(raw || '')
    .split(/[,\n;/]+/)
    .map((chunk) => chunk.trim())
    .filter(Boolean)
}

export function uniqueList(values) {
  return Array.from(new Set(values))
}

async function parseError(response) {
  try {
    const payload = await response.json()
    if (payload && typeof payload.detail === 'string') {
      return payload.detail
    }
  } catch {
    // Ignore JSON parse errors for API error handling.
  }
  return `Request failed (${response.status})`
}

function escapePdfText(value) {
  return String(value || '')
    .replace(/\\/g, '\\\\')
    .replace(/\(/g, '\\(')
    .replace(/\)/g, '\\)')
}

function wrapText(text, maxChars = 78) {
  const words = String(text || '').split(/\s+/).filter(Boolean)
  if (words.length === 0) {
    return []
  }

  const lines = []
  let current = words[0]
  for (let index = 1; index < words.length; index += 1) {
    const next = words[index]
    if (`${current} ${next}`.length <= maxChars) {
      current = `${current} ${next}`
    } else {
      lines.push(current)
      current = next
    }
  }
  lines.push(current)
  return lines
}

async function loadLogoAsJpeg(src, maxEdge = 96) {
  const image = await new Promise((resolve, reject) => {
    const element = new Image()
    element.onload = () => resolve(element)
    element.onerror = () => reject(new Error('Unable to load logo image for PDF export.'))
    element.src = src
  })

  const scale = Math.min(maxEdge / image.width, maxEdge / image.height, 1)
  const width = Math.max(1, Math.round(image.width * scale))
  const height = Math.max(1, Math.round(image.height * scale))

  const canvas = document.createElement('canvas')
  canvas.width = width
  canvas.height = height

  const context = canvas.getContext('2d')
  if (!context) {
    throw new Error('Unable to prepare PDF report image context.')
  }

  context.fillStyle = '#ffffff'
  context.fillRect(0, 0, width, height)
  context.drawImage(image, 0, 0, width, height)

  const dataUrl = canvas.toDataURL('image/jpeg', 0.9)
  const base64 = dataUrl.split(',')[1] || ''
  const binary = atob(base64)
  return { width, height, binary }
}

// function buildFinalReportPdf({
//   logo,
//   diagnosis,
//   confidence,
//   severity,
//   recommendation,
//   medicalNotice,
//   generatedAt,
// }) {
//   const pageWidth = 595
//   const pageHeight = 842
//   const left = 54
//   const right = pageWidth - 54
//   const contentWidth = right - left

//   const commands = []
//   commands.push('1 1 1 rg')
//   commands.push(`0 0 ${pageWidth} ${pageHeight} re f`)

//   commands.push('0.95 0.98 1 rg')
//   commands.push(`${left} ${pageHeight - 156} ${contentWidth} 108 re f`)
//   commands.push('0.81 0.89 0.97 RG')
//   commands.push('1.1 w')
//   commands.push(`${left} ${pageHeight - 156} ${contentWidth} 108 re S`)

//   const logoDrawWidth = 62
//   const logoDrawHeight = 62
//   const logoY = pageHeight - 132
//   commands.push(`q ${logoDrawWidth} 0 0 ${logoDrawHeight} ${left + 14} ${logoY} cm /Im1 Do Q`)

//   const headerX = left + 92
//   commands.push(`BT /F2 24 Tf 0.08 0.26 0.48 rg ${headerX} ${pageHeight - 86} Td (Skinova AI) Tj ET`)
//   commands.push(`BT /F2 13 Tf 0.17 0.36 0.58 rg ${headerX} ${pageHeight - 108} Td (Final Diagnosis Report) Tj ET`)
//   commands.push(`BT /F1 10 Tf 0.35 0.48 0.62 rg ${headerX} ${pageHeight - 124} Td (${escapePdfText(generatedAt)}) Tj ET`)

//   const sectionInset = 16
//   const sectionWidth = contentWidth - sectionInset * 2

//   const diagnosisBox = { y: 528, h: 162 }
//   const recommendationBox = { y: 314, h: 194 }
//   const noticeBox = { y: 180, h: 108 }

//   commands.push('0.98 0.99 1 rg')
//   commands.push(`${left + sectionInset} ${diagnosisBox.y} ${sectionWidth} ${diagnosisBox.h} re f`)
//   commands.push(`${left + sectionInset} ${recommendationBox.y} ${sectionWidth} ${recommendationBox.h} re f`)
//   commands.push(`${left + sectionInset} ${noticeBox.y} ${sectionWidth} ${noticeBox.h} re f`)
//   commands.push('0.84 0.9 0.97 RG')
//   commands.push('1 w')
//   commands.push(`${left + sectionInset} ${diagnosisBox.y} ${sectionWidth} ${diagnosisBox.h} re S`)
//   commands.push(`${left + sectionInset} ${recommendationBox.y} ${sectionWidth} ${recommendationBox.h} re S`)
//   commands.push(`${left + sectionInset} ${noticeBox.y} ${sectionWidth} ${noticeBox.h} re S`)

//   const textLeft = left + sectionInset + 18

//   commands.push(`BT /F2 14 Tf 0.1 0.28 0.49 rg ${textLeft} ${diagnosisBox.y + diagnosisBox.h - 28} Td (Final Diagnosis) Tj ET`)
//   commands.push(`BT /F2 23 Tf 0.07 0.24 0.43 rg ${textLeft} ${diagnosisBox.y + diagnosisBox.h - 60} Td (${escapePdfText(diagnosis)}) Tj ET`)
//   commands.push(
//     `BT /F1 12 Tf 0.23 0.35 0.52 rg ${textLeft} ${diagnosisBox.y + diagnosisBox.h - 86} Td (Confidence: ${escapePdfText(confidence)}) Tj ET`
//   )
//   commands.push(
//     `BT /F1 12 Tf 0.23 0.35 0.52 rg ${textLeft} ${diagnosisBox.y + diagnosisBox.h - 104} Td (Severity: ${escapePdfText(severity)}) Tj ET`
//   )

//   commands.push(
//     `BT /F2 14 Tf 0.1 0.28 0.49 rg ${textLeft} ${recommendationBox.y + recommendationBox.h - 28} Td (Recommendation) Tj ET`
//   )
//   const recommendationLines = wrapText(recommendation, 80)
//   const maxRecommendationLines = 9
//   let recommendationY = recommendationBox.y + recommendationBox.h - 52
//   for (const [index, line] of recommendationLines.entries()) {
//     if (index >= maxRecommendationLines) {
//       break
//     }
//     const lastVisible = index === maxRecommendationLines - 1 && recommendationLines.length > maxRecommendationLines
//     const displayLine = lastVisible ? `${line.slice(0, 72)}...` : line
//     commands.push(`BT /F1 11 Tf 0.26 0.38 0.55 rg ${textLeft} ${recommendationY} Td (${escapePdfText(displayLine)}) Tj ET`)
//     recommendationY -= 16
//   }

//   commands.push(`BT /F2 13 Tf 0.19 0.31 0.48 rg ${textLeft} ${noticeBox.y + noticeBox.h - 28} Td (Medical Notice) Tj ET`)
//   const noticeLines = wrapText(medicalNotice, 84)
//   let noticeY = noticeBox.y + noticeBox.h - 50
//   for (const line of noticeLines.slice(0, 3)) {
//     commands.push(`BT /F1 10 Tf 0.34 0.45 0.6 rg ${textLeft} ${noticeY} Td (${escapePdfText(line)}) Tj ET`)
//     noticeY -= 14
//   }

//   commands.push('0.86 0.92 0.98 RG')
//   commands.push('0.8 w')
//   commands.push(`${left + 6} 126 ${right - left - 12} 0 m ${right - 6} 126 l S`)
//   commands.push(
//     `BT /F1 9 Tf 0.48 0.57 0.67 rg ${left + 12} 108 Td (Generated by Skinova AI for decision-support documentation.) Tj ET`
//   )

//   const content = commands.join('\n')

//   const objects = [
//     '1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n',
//     '2 0 obj\n<< /Type /Pages /Count 1 /Kids [3 0 R] >>\nendobj\n',
//     `3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 ${pageWidth} ${pageHeight}] /Resources << /Font << /F1 4 0 R /F2 5 0 R >> /XObject << /Im1 6 0 R >> >> /Contents 7 0 R >>\nendobj\n`,
//     '4 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n',
//     '5 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold >>\nendobj\n',
//     `6 0 obj\n<< /Type /XObject /Subtype /Image /Width ${logo.width} /Height ${logo.height} /ColorSpace /DeviceRGB /BitsPerComponent 8 /Filter /DCTDecode /Length ${logo.binary.length} >>\nstream\n${logo.binary}\nendstream\nendobj\n`,
//     `7 0 obj\n<< /Length ${content.length} >>\nstream\n${content}\nendstream\nendobj\n`,
//   ]

//   let pdf = '%PDF-1.4\n'
//   const offsets = [0]
//   for (const object of objects) {
//     offsets.push(pdf.length)
//     pdf += object
//   }

//   const xrefStart = pdf.length
//   pdf += `xref\n0 ${objects.length + 1}\n`
//   pdf += '0000000000 65535 f \n'
//   for (let index = 1; index <= objects.length; index += 1) {
//     pdf += `${String(offsets[index]).padStart(10, '0')} 00000 n \n`
//   }
//   pdf += `trailer\n<< /Size ${objects.length + 1} /Root 1 0 R >>\nstartxref\n${xrefStart}\n%%EOF`

//   const bytes = new Uint8Array(pdf.length)
//   for (let index = 0; index < pdf.length; index += 1) {
//     bytes[index] = pdf.charCodeAt(index) & 0xff
//   }
//   return bytes
// }

function buildFinalReportPdf({
  logo,
  diagnosis,
  confidence,
  severity,
  recommendation,
  medicalNotice,
  generatedAt,
}) {
  const pageWidth = 595
  const pageHeight = 842

  const marginX = 54
  const topY = pageHeight - 54
  const bottomY = 54
  const contentWidth = pageWidth - marginX * 2

  const headerHeight = 108
  const sectionGap = 18
  const innerPad = 18

  const diagnosisBox = {
    x: marginX,
    y: 524,
    w: contentWidth,
    h: 154,
  }

  const recommendationBox = {
    x: marginX,
    y: 300,
    w: contentWidth,
    h: 198,
  }

  const noticeBox = {
    x: marginX,
    y: 168,
    w: contentWidth,
    h: 100,
  }

  const headerBox = {
    x: marginX,
    y: pageHeight - 162,
    w: contentWidth,
    h: headerHeight,
  }

  const commands = []

  const drawFilledBox = (x, y, w, h, fillRgb, strokeRgb, lineWidth = 1) => {
    commands.push(`${fillRgb} rg`)
    commands.push(`${x} ${y} ${w} ${h} re f`)
    commands.push(`${strokeRgb} RG`)
    commands.push(`${lineWidth} w`)
    commands.push(`${x} ${y} ${w} ${h} re S`)
  }

  const text = (font, size, rgb, x, y, value) => {
    commands.push(`BT /${font} ${size} Tf ${rgb} rg ${x} ${y} Td (${escapePdfText(value)}) Tj ET`)
  }

  const centerText = (font, size, rgb, centerX, y, value) => {
    const width = estimateTextWidth(value, size, font === 'F2' ? 0.56 : 0.52)
    const x = centerX - width / 2
    text(font, size, rgb, x, y, value)
  }

  const wrapTextByWidth = (input, maxWidth, fontSize, factor = 0.52) => {
    const words = String(input || '').split(/\s+/).filter(Boolean)
    const lines = []
    let line = ''

    for (const word of words) {
      const trial = line ? `${line} ${word}` : word
      if (estimateTextWidth(trial, fontSize, factor) <= maxWidth) {
        line = trial
      } else {
        if (line) lines.push(line)
        line = word
      }
    }
    if (line) lines.push(line)
    return lines
  }

  const fitLineWithEllipsis = (value, maxWidth, fontSize, factor = 0.52) => {
    if (estimateTextWidth(value, fontSize, factor) <= maxWidth) return value
    let out = value
    while (out.length > 0 && estimateTextWidth(`${out}...`, fontSize, factor) > maxWidth) {
      out = out.slice(0, -1)
    }
    return `${out}...`
  }

  // Background
  commands.push('1 1 1 rg')
  commands.push(`0 0 ${pageWidth} ${pageHeight} re f`)

  // Header
  drawFilledBox(
    headerBox.x,
    headerBox.y,
    headerBox.w,
    headerBox.h,
    '0.95 0.98 1',
    '0.81 0.89 0.97',
    1.1
  )

  const logoSize = 62
  const logoX = headerBox.x + 18
  const logoY = headerBox.y + (headerBox.h - logoSize) / 2
  commands.push(`q ${logoSize} 0 0 ${logoSize} ${logoX} ${logoY} cm /Im1 Do Q`)

  const headerTextX = logoX + logoSize + 20
  const headerRightPad = 18
  const headerTextWidth = headerBox.x + headerBox.w - headerRightPad - headerTextX

  text('F2', 24, '0.08 0.26 0.48', headerTextX, headerBox.y + 70, 'Skinova AI')
  text('F2', 13, '0.17 0.36 0.58', headerTextX, headerBox.y + 46, 'Final Diagnosis Report')
  text('F1', 10, '0.35 0.48 0.62', headerTextX, headerBox.y + 28, generatedAt)

  // Section boxes
  drawFilledBox(
    diagnosisBox.x,
    diagnosisBox.y,
    diagnosisBox.w,
    diagnosisBox.h,
    '0.98 0.99 1',
    '0.84 0.9 0.97'
  )

  drawFilledBox(
    recommendationBox.x,
    recommendationBox.y,
    recommendationBox.w,
    recommendationBox.h,
    '0.98 0.99 1',
    '0.84 0.9 0.97'
  )

  drawFilledBox(
    noticeBox.x,
    noticeBox.y,
    noticeBox.w,
    noticeBox.h,
    '0.98 0.99 1',
    '0.84 0.9 0.97'
  )

  const sectionTextX = marginX + innerPad
  const lineMaxWidth = contentWidth - innerPad * 2

  // Diagnosis section
  text('F2', 14, '0.1 0.28 0.49', sectionTextX, diagnosisBox.y + diagnosisBox.h - 28, 'Final Diagnosis')

  const diagnosisDisplay = fitLineWithEllipsis(diagnosis, lineMaxWidth, 22, 0.56)
  text('F2', 22, '0.07 0.24 0.43', sectionTextX, diagnosisBox.y + diagnosisBox.h - 60, diagnosisDisplay)

  text(
    'F1',
    12,
    '0.23 0.35 0.52',
    sectionTextX,
    diagnosisBox.y + diagnosisBox.h - 88,
    `Confidence: ${confidence}`
  )

  text(
    'F1',
    12,
    '0.23 0.35 0.52',
    sectionTextX,
    diagnosisBox.y + diagnosisBox.h - 108,
    `Severity: ${severity}`
  )

  // Recommendation section
  text(
    'F2',
    14,
    '0.1 0.28 0.49',
    sectionTextX,
    recommendationBox.y + recommendationBox.h - 28,
    'Recommendation'
  )

  const recommendationLines = wrapTextByWidth(recommendation, lineMaxWidth, 11, 0.52)
  const recLineHeight = 16
  const recTopY = recommendationBox.y + recommendationBox.h - 54
  const recMaxLines = Math.floor((recommendationBox.h - 70) / recLineHeight)

  for (let i = 0; i < Math.min(recommendationLines.length, recMaxLines); i += 1) {
    const isLastVisible = i === recMaxLines - 1 && recommendationLines.length > recMaxLines
    const line = isLastVisible
      ? fitLineWithEllipsis(recommendationLines[i], lineMaxWidth, 11, 0.52)
      : recommendationLines[i]

    text('F1', 11, '0.26 0.38 0.55', sectionTextX, recTopY - i * recLineHeight, line)
  }

  // Notice section
  text(
    'F2',
    13,
    '0.19 0.31 0.48',
    sectionTextX,
    noticeBox.y + noticeBox.h - 26,
    'Medical Notice'
  )

  const noticeLines = wrapTextByWidth(medicalNotice, lineMaxWidth, 10, 0.5)
  const noticeLineHeight = 14
  const noticeTopY = noticeBox.y + noticeBox.h - 48
  const noticeMaxLines = 3

  for (let i = 0; i < Math.min(noticeLines.length, noticeMaxLines); i += 1) {
    const isLastVisible = i === noticeMaxLines - 1 && noticeLines.length > noticeMaxLines
    const line = isLastVisible
      ? fitLineWithEllipsis(noticeLines[i], lineMaxWidth, 10, 0.5)
      : noticeLines[i]

    text('F1', 10, '0.34 0.45 0.6', sectionTextX, noticeTopY - i * noticeLineHeight, line)
  }

  // Footer
  commands.push('0.86 0.92 0.98 RG')
  commands.push('0.8 w')
  commands.push(`${marginX} 124 ${pageWidth - marginX * 2} 0 m ${pageWidth - marginX} 124 l S`)

  centerText(
    'F1',
    9,
    '0.48 0.57 0.67',
    pageWidth / 2,
    106,
    'Generated by Skinova AI for decision-support documentation.'
  )

  const content = commands.join('\n')

  const objects = [
    '1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n',
    '2 0 obj\n<< /Type /Pages /Count 1 /Kids [3 0 R] >>\nendobj\n',
    `3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 ${pageWidth} ${pageHeight}] /Resources << /Font << /F1 4 0 R /F2 5 0 R >> /XObject << /Im1 6 0 R >> >> /Contents 7 0 R >>\nendobj\n`,
    '4 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n',
    '5 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold >>\nendobj\n',
    `6 0 obj\n<< /Type /XObject /Subtype /Image /Width ${logo.width} /Height ${logo.height} /ColorSpace /DeviceRGB /BitsPerComponent 8 /Filter /DCTDecode /Length ${logo.binary.length} >>\nstream\n${logo.binary}\nendstream\nendobj\n`,
    `7 0 obj\n<< /Length ${content.length} >>\nstream\n${content}\nendstream\nendobj\n`,
  ]

  let pdf = '%PDF-1.4\n'
  const offsets = [0]

  for (const object of objects) {
    offsets.push(pdf.length)
    pdf += object
  }

  const xrefStart = pdf.length
  pdf += `xref\n0 ${objects.length + 1}\n`
  pdf += '0000000000 65535 f \n'

  for (let i = 1; i <= objects.length; i += 1) {
    pdf += `${String(offsets[i]).padStart(10, '0')} 00000 n \n`
  }

  pdf += `trailer\n<< /Size ${objects.length + 1} /Root 1 0 R >>\nstartxref\n${xrefStart}\n%%EOF`

  const bytes = new Uint8Array(pdf.length)
  for (let i = 0; i < pdf.length; i += 1) {
    bytes[i] = pdf.charCodeAt(i) & 0xff
  }

  return bytes
}

function estimateTextWidth(value, fontSize, factor = 0.52) {
  return String(value || '').length * fontSize * factor
}

export default function App() {
  const [started, setStarted] = useState(false)

  const [file, setFile] = useState(null)
  const [previewUrl, setPreviewUrl] = useState('')

  const [predicting, setPredicting] = useState(false)
  const [predictError, setPredictError] = useState('')
  const [predictResult, setPredictResult] = useState(null)

  const [catalogLoading, setCatalogLoading] = useState(false)
  const [catalog, setCatalog] = useState([])
  const [suggestions, setSuggestions] = useState([])
  const [showSuggestions, setShowSuggestions] = useState(false)

  const [symptomQuery, setSymptomQuery] = useState('')
  const [selectedSymptoms, setSelectedSymptoms] = useState([])

  const [keywordInput, setKeywordInput] = useState('')
  const [customKeywords, setCustomKeywords] = useState([])

  const [finalizing, setFinalizing] = useState(false)
  const [finalError, setFinalError] = useState('')
  const [finalResult, setFinalResult] = useState(null)
  const [downloadingReport, setDownloadingReport] = useState(false)

  useEffect(() => {
    let ignore = false

    async function loadCatalog() {
      setCatalogLoading(true)
      try {
        const response = await fetch(`${API_BASE}/symptoms/catalog`)
        if (!response.ok) {
          throw new Error(await parseError(response))
        }
        const payload = await response.json()
        if (!ignore) {
          setCatalog(Array.isArray(payload.symptoms) ? payload.symptoms : [])
        }
      } catch {
        if (!ignore) {
          setCatalog([])
        }
      } finally {
        if (!ignore) {
          setCatalogLoading(false)
        }
      }
    }

    loadCatalog()
    return () => {
      ignore = true
    }
  }, [])

  useEffect(() => {
    return () => {
      if (previewUrl) {
        URL.revokeObjectURL(previewUrl)
      }
    }
  }, [previewUrl])

  const symptomLabelMap = useMemo(() => {
    const map = new Map()
    for (const row of catalog) {
      map.set(row.name, row.label)
    }
    for (const row of suggestions) {
      if (!map.has(row.name)) {
        map.set(row.name, row.label)
      }
    }
    return map
  }, [catalog, suggestions])

  const filteredCatalog = useMemo(() => {
    const query = symptomQuery.trim().toLowerCase()
    if (!query) {
      return catalog.slice(0, 32)
    }
    return catalog
      .filter((row) => {
        const label = String(row.label || '').toLowerCase()
        const name = String(row.name || '').toLowerCase()
        return label.includes(query) || name.includes(query)
      })
      .slice(0, 32)
  }, [catalog, symptomQuery])

  const topPredictions = useMemo(() => {
    if (!predictResult || !Array.isArray(predictResult.topk)) {
      return []
    }
    return predictResult.topk.slice(0, 5)
  }, [predictResult])

  const suggestionSeedTopk = useMemo(() => topPredictions.slice(0, 3), [topPredictions])

  const pendingKeywordTerms = useMemo(() => splitKeywordText(keywordInput), [keywordInput])
  const effectiveKeywords = useMemo(
    () => uniqueList([...customKeywords, ...pendingKeywordTerms]),
    [customKeywords, pendingKeywordTerms]
  )

  function resetRefinementState() {
    setSuggestions([])
    setShowSuggestions(false)
    setSymptomQuery('')
    setSelectedSymptoms([])
    setKeywordInput('')
    setCustomKeywords([])
    setFinalError('')
    setFinalResult(null)
  }

  function onFileChange(event) {
    const chosen = event.target.files?.[0] || null
    if (previewUrl) {
      URL.revokeObjectURL(previewUrl)
    }
    setFile(chosen)
    setPreviewUrl(chosen ? URL.createObjectURL(chosen) : '')
    setPredictError('')
    setPredictResult(null)
    resetRefinementState()
  }

  function toggleSymptom(symptomName) {
    setSelectedSymptoms((previous) => {
      if (previous.includes(symptomName)) {
        return previous.filter((name) => name !== symptomName)
      }
      return [...previous, symptomName]
    })
    setFinalResult(null)
    setFinalError('')
  }

  function addKeywordTerms(raw) {
    const terms = splitKeywordText(raw)
    if (terms.length === 0) {
      return
    }
    setCustomKeywords((previous) => uniqueList([...previous, ...terms]))
    setKeywordInput('')
    setFinalResult(null)
    setFinalError('')
  }

  function removeKeyword(term) {
    setCustomKeywords((previous) => previous.filter((item) => item !== term))
    setFinalResult(null)
    setFinalError('')
  }

  async function handlePredict(event) {
    event.preventDefault()

    if (!file) {
      setPredictError('Select an image file before prediction.')
      return
    }

    setPredicting(true)
    setPredictError('')
    setPredictResult(null)
    resetRefinementState()

    try {
      const formData = new FormData()
      formData.append('file', file)

      const predictResponse = await fetch(`${API_BASE}/predict?topk=5`, {
        method: 'POST',
        body: formData,
      })
      if (!predictResponse.ok) {
        throw new Error(await parseError(predictResponse))
      }

      const prediction = await predictResponse.json()
      setPredictResult(prediction)

      const topRows = Array.isArray(prediction.topk) ? prediction.topk.slice(0, 5) : []
      if (hasHighConfidenceNeutral(topRows)) {
        setShowSuggestions(false)
        setSuggestions([])
        return
      }

      const suggestResponse = await fetch(`${API_BASE}/symptoms/suggest`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          topk: topRows.slice(0, 3),
          limit: 5,
        }),
      })

      if (suggestResponse.ok) {
        const payload = await suggestResponse.json()
        setShowSuggestions(payload.show_suggestions !== false)
        setSuggestions(Array.isArray(payload.suggestions) ? payload.suggestions : [])
        if (Array.isArray(payload.symptoms) && payload.symptoms.length > 0) {
          setCatalog(payload.symptoms)
        }
      } else {
        setShowSuggestions(false)
        setSuggestions([])
      }
    } catch (error) {
      setPredictError(error.message || 'Unable to run prediction.')
    } finally {
      setPredicting(false)
    }
  }

  async function handleFinalPredict() {
    if (selectedSymptoms.length === 0 && effectiveKeywords.length === 0) {
      setFinalError('Select symptoms or enter symptom keywords before final prediction.')
      return
    }

    setFinalizing(true)
    setFinalError('')
    setFinalResult(null)

    try {
      const response = await fetch(`${API_BASE}/symptoms/final-predict`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          selected_symptoms: selectedSymptoms,
          custom_keywords: effectiveKeywords,
          image_topk: topPredictions,
          limit: 5,
        }),
      })

      if (!response.ok) {
        throw new Error(await parseError(response))
      }
      const payload = await response.json()
      setFinalResult(payload)
    } catch (error) {
      setFinalError(error.message || 'Unable to compute final prediction.')
    } finally {
      setFinalizing(false)
    }
  }

  async function handleDownloadReport() {
    if (!finalResult?.final_prediction) {
      return
    }

    setDownloadingReport(true)
    try {
      const logo = await loadLogoAsJpeg(logoImage)
      const pdfBytes = buildFinalReportPdf({
        logo,
        diagnosis: toLabel(finalResult.final_prediction?.diagnosis || 'Unknown'),
        confidence: formatPercent(finalResult.final_prediction?.confidence || 0),
        severity: toLabel(finalResult.final_prediction?.severity_level || 'Unknown'),
        recommendation: finalResult.final_prediction?.treatment_recommendation || 'Consult a qualified clinician.',
        medicalNotice: 'This is decision-support output, not medical advice.',
        generatedAt: `Generated: ${new Date().toLocaleString()}`,
      })

      const blob = new Blob([pdfBytes], { type: 'application/pdf' })
      const url = URL.createObjectURL(blob)
      const anchor = document.createElement('a')
      anchor.href = url
      anchor.download = `skinova-report-${Date.now()}.pdf`
      document.body.appendChild(anchor)
      anchor.click()
      anchor.remove()
      URL.revokeObjectURL(url)
    } catch {
      setFinalError('Unable to generate PDF report. Please try again.')
    } finally {
      setDownloadingReport(false)
    }
  }

  if (!started) {
    return (
      <div className="landing-page">
        <section className="hero-shell">
          <header className="hero-topbar">
            <div className="hero-brand">
              <div className="hero-brand-icon" aria-hidden="true">
                <img src={logoImage} alt="" />
              </div>
              <div>
                <p className="hero-brand-title">Skinova AI</p>
                <p className="hero-brand-sub">Skin Scanner</p>
              </div>
            </div>
            {/* <nav className="hero-nav" aria-label="Landing quick links">
              <a href="#how-it-works">How it works</a>
              <a href="#why-skinova">Benefits</a>
              <a href="#workflow-deep">Workflow</a>
              <a href="#faq">FAQ</a>
              <a href="#medical-note">Medical notice</a>
            </nav> */}
            {/* <button type="button" className="btn hero-mini-cta" onClick={() => setStarted(true)}>
              Start Scan
            </button> */}
          </header>

          <div className="hero-content">
            <div className="hero-left">
              {/* <p className="hero-kicker">AI screening support for home and clinic triage</p> */}
              <h1>Check your skin with a faster, clearer workflow.</h1>
              <p className="hero-lead">
                Upload one image, review AI top-5 suggestions, refine with symptom context, and generate a
                clinician-ready result in minutes.
              </p>

              <div className="hero-stats">
                <article className="hero-stat">
                  <p className="hero-stat-value">Top-5</p>
                  <p className="hero-stat-label">Disease ranking from one image</p>
                </article>
                <article className="hero-stat">
                  <p className="hero-stat-value">2-Step</p>
                  <p className="hero-stat-label">Image + symptom refinement</p>
                </article>
                <article className="hero-stat">
                  <p className="hero-stat-value">Private</p>
                  <p className="hero-stat-label">Local workflow, no account required</p>
                </article>
              </div>

              <div className="hero-steps" id="how-it-works">
                <article className="hero-step">
                  <div className="hero-step-icon cam">CAM</div>
                  <div>
                    <span className="hero-step-tag">STEP 1</span>
                    <p>Take a photo of a skin concern</p>
                  </div>
                </article>
                <article className="hero-step">
                  <div className="hero-step-icon ai">AI</div>
                  <div>
                    <span className="hero-step-tag">STEP 2</span>
                    <p>AI instantly analyzes your photo</p>
                  </div>
                </article>
                <article className="hero-step">
                  <div className="hero-step-icon pdf">PDF</div>
                  <div>
                    <span className="hero-step-tag">STEP 3</span>
                    <p>Get a personalized PDF report</p>
                  </div>
                </article>
                <article className="hero-step">
                  <div className="hero-step-icon care">CARE</div>
                  <div>
                    <span className="hero-step-tag">STEP 4</span>
                    <p>AI consultant explains your result</p>
                  </div>
                </article>
              </div>

              <div className="hero-action-row">
                <button type="button" className="hero-cta" onClick={() => setStarted(true)}>
                  GET INSTANT RESULT
                </button>
                <p className="hero-action-note">No sign-up required</p>
              </div>
              <p className="hero-note" id="medical-note">
                * The scan result is not a diagnosis. For accurate diagnosis and treatment, consult a clinician.
              </p>
            </div>

            <div className="hero-right" aria-hidden="true">
              <div className="hero-right-body" />
              <div className="scan-focus">
                <div className="scan-target">
                  <div className="scan-dot" />
                </div>
                <p>
                  Take a photo of a skin lesion
                  <br />
                  and receive risk assessment
                </p>
              </div>
              <article className="hero-preview-card">
                <p className="hero-preview-title">Session Flow</p>
                <p className="hero-preview-item">1. Upload skin image</p>
                <p className="hero-preview-item">2. Review AI top-5</p>
                <p className="hero-preview-item">3. Add symptoms & finalize</p>
                <p className="hero-preview-foot">Designed for faster decision support.</p>
              </article>
            </div>
          </div>
        </section>

        <section className="landing-why" id="why-skinova">
          <h2>Why choose Skinova AI?</h2>
          <p>Built for practical screening support with a clear, clinician-first interaction model.</p>
          <div className="landing-why-grid">
            <article className="landing-why-card">
              <p className="landing-why-card-title">Fast triage</p>
              <p>Get ranked differential results quickly from a single image upload.</p>
            </article>
            <article className="landing-why-card">
              <p className="landing-why-card-title">Symptom-aware refinement</p>
              <p>Combine image evidence and symptom selection for stronger final predictions.</p>
            </article>
            <article className="landing-why-card">
              <p className="landing-why-card-title">Decision-ready outputs</p>
              <p>Generate final confidence and recommendation text for next-step guidance.</p>
            </article>
          </div>
        </section>

        <section className="landing-detail detail-workflow" id="workflow-deep">
          <div className="detail-head">
            <p className="detail-kicker">Detailed Workflow</p>
            <h2>A complete screening journey from upload to final recommendation.</h2>
            <p>
              Skinova is designed as a guided workflow. Each phase adds clarity so users can move from uncertainty
              to an actionable next step with confidence.
            </p>
          </div>
          <div className="detail-layout">
            <div className="timeline-grid">
              <article className="timeline-card">
                <span>Phase 01</span>
                <h3>Capture & Upload</h3>
                <p>Use a clear, well-lit image to begin the initial visual assessment pipeline.</p>
              </article>
              <article className="timeline-card">
                <span>Phase 02</span>
                <h3>Image Top-5 Ranking</h3>
                <p>AI returns ranked disease probabilities with confidence bars for transparent review.</p>
              </article>
              <article className="timeline-card">
                <span>Phase 03</span>
                <h3>Symptom Refinement</h3>
                <p>Add clinically relevant symptoms or custom terms to narrow possible diagnoses.</p>
              </article>
              <article className="timeline-card">
                <span>Phase 04</span>
                <h3>Final Result Output</h3>
                <p>Receive a final prediction, severity tag, and recommendation text for next-step triage.</p>
              </article>
            </div>

            <aside className="detail-panel">
              <h3>Best Result Tips</h3>
              <ul>
                <li>Take photos in natural light where lesion borders are visible.</li>
                <li>Avoid blur, heavy filters, or compressed screenshots.</li>
                <li>Add symptom details like itching, pain, redness, or scaling.</li>
                <li>Use repeated scans to monitor progression over time.</li>
              </ul>
            </aside>
          </div>
        </section>

        <section className="landing-detail detail-report" id="report-details">
          <div className="detail-head">
            <p className="detail-kicker">Report Coverage</p>
            <h2>What the final screening output includes.</h2>
            <p>
              The interface is built to make results easy to read and discuss with a clinician or care team.
            </p>
          </div>
          <div className="report-grid">
            <article className="report-card">
              <h3>Primary Diagnosis</h3>
              <p>Main predicted condition based on image and symptom context.</p>
            </article>
            <article className="report-card">
              <h3>Differential Top-5</h3>
              <p>Alternative conditions with confidence percentages for comparison.</p>
            </article>
            <article className="report-card">
              <h3>Severity Signal</h3>
              <p>Quick severity level to support urgency and follow-up prioritization.</p>
            </article>
            <article className="report-card">
              <h3>Treatment Guidance Text</h3>
              <p>Recommendation snippet to guide safe next actions before consultation.</p>
            </article>
            <article className="report-card">
              <h3>Symptom Influence</h3>
              <p>Shows how symptom choices refine the final diagnostic direction.</p>
            </article>
            <article className="report-card">
              <h3>Medical Notice</h3>
              <p>Always reminds users that final medical diagnosis requires clinicians.</p>
            </article>
          </div>
        </section>

        <section className="landing-detail detail-audience" id="for-whom">
          <div className="detail-head">
            <p className="detail-kicker">Who It Supports</p>
            <h2>Designed for both self-checking and clinic-facing screening workflows.</h2>
          </div>
          <div className="audience-grid">
            <article className="audience-card">
              <h3>Individuals & Families</h3>
              <p>Get early risk signals at home before deciding when to seek care.</p>
            </article>
            <article className="audience-card">
              <h3>Primary Care Teams</h3>
              <p>Use quick image-first triage support when dermatology access is limited.</p>
            </article>
            <article className="audience-card">
              <h3>Community Screenings</h3>
              <p>Standardize first-pass visual assessment across outreach programs.</p>
            </article>
          </div>
        </section>

        <section className="landing-detail detail-faq" id="faq">
          <div className="detail-head">
            <p className="detail-kicker">Frequently Asked Questions</p>
            <h2>Common questions before using the scanner.</h2>
          </div>
          <div className="faq-list">
            <details>
              <summary>Is this a final diagnosis?</summary>
              <p>
                No. Skinova is a decision-support tool. Use it for screening guidance, then confirm with a qualified
                clinician.
              </p>
            </details>
            <details>
              <summary>Can I use it with any skin photo?</summary>
              <p>
                Yes, but image quality matters. Clear lighting and focused lesion visibility improve reliability.
              </p>
            </details>
            <details>
              <summary>Why should I add symptoms after image prediction?</summary>
              <p>
                Symptom refinement gives context that can shift confidence and improve final ranking accuracy.
              </p>
            </details>
            <details>
              <summary>What if a healthy/none result appears at high confidence?</summary>
              <p>
                The app hides symptom suggestions in that case to avoid unnecessary over-refinement and keeps output
                concise.
              </p>
            </details>
            <details>
              <summary>Who should avoid self-triage and seek immediate care?</summary>
              <p>
                Users with rapidly worsening lesions, fever, severe pain, or signs of infection should seek urgent
                medical evaluation.
              </p>
            </details>
          </div>
        </section>

        <section className="landing-cta-band">
          <div>
            <p className="detail-kicker">Ready To Start</p>
            <h2>Begin your screening session now.</h2>
            <p>Upload an image and follow the guided workflow for a complete result view.</p>
          </div>
          <button type="button" className="hero-cta" onClick={() => setStarted(true)}>
            START SCREENING
          </button>
        </section>
      </div>
    )
  }

  return (
    <div className="app-page">
      <header className="topbar">
        <div className="brand">
          <img src={logoImage} alt="" className="brand-dot" aria-hidden="true" />
          <span>Skinova AI</span>
        </div>
        <button type="button" className="btn ghost" onClick={() => setStarted(false)}>
          Back To Hero
        </button>
      </header>

      <main className="dashboard">
        <section className="panel upload-card">
          <h2>Image Scan</h2>
          <p className="subtitle">Upload a skin image to generate top-5 disease predictions.</p>

          <form className="upload-form" onSubmit={handlePredict}>
            <label className="file-input-wrap">
              <span>Select image</span>
              <input type="file" accept="image/*" onChange={onFileChange} />
            </label>
            <div className="file-name">{file ? file.name : 'No file selected'}</div>
            <button className="btn primary" type="submit" disabled={predicting || !file}>
              {predicting ? 'Predicting...' : 'Run Prediction'}
            </button>
          </form>

          {predictError ? <p className="error-text">{predictError}</p> : null}

          {previewUrl ? (
            <div className="preview">
              <img src={previewUrl} alt="Selected skin preview" />
            </div>
          ) : (
            <div className="empty-box">Image preview appears here.</div>
          )}
        </section>

        <section className="panel result-card">
          <h2>Top-5 Result</h2>
          {!predictResult ? (
            <div className="empty-box">Run prediction to view top-5 disease probabilities.</div>
          ) : (
            <>
              <div className="result-summary">
                <p>
                  Predicted disease: <strong>{toLabel(predictResult.prediction)}</strong>
                </p>
                <p>
                  Confidence: <strong>{formatPercent(predictResult.confidence)}</strong>
                </p>
                <p>
                  Inference: <strong>FedAvg Global (4 clients)</strong>
                </p>
              </div>

              <ul className="score-list">
                {topPredictions.map((row, index) => {
                  const width = `${(confidenceToFraction(row.confidence) * 100).toFixed(2)}%`
                  return (
                    <li key={`${row.prediction}-${index}`}>
                      <div className="score-head">
                        <span>{toLabel(row.prediction)}</span>
                        <span>{formatPercent(row.confidence)}</span>
                      </div>
                      <div className="bar-track">
                        <div className="bar-fill" style={{ width }} />
                      </div>
                    </li>
                  )
                })}
              </ul>
            </>
          )}
        </section>

        <section className="panel symptom-card">
          <h2>Symptom Refinement</h2>
          <p className="subtitle">
            Suggestions come from top-3 image diseases. Hidden when None/Healthy appears with 80%+ confidence.
          </p>

          {showSuggestions && suggestions.length > 0 ? (
            <div className="chip-block">
              <h3>Suggested Symptoms</h3>
              <div className="chip-grid">
                {suggestions.map((item) => {
                  const active = selectedSymptoms.includes(item.name)
                  return (
                    <button
                      key={item.name}
                      type="button"
                      className={`chip ${active ? 'active' : ''}`}
                      onClick={() => toggleSymptom(item.name)}
                    >
                      {item.label}
                    </button>
                  )
                })}
              </div>
              {suggestionSeedTopk.length > 0 ? (
                <p className="hint-text">
                  Source diseases: {suggestionSeedTopk.map((row) => toLabel(row.prediction)).join(', ')}
                </p>
              ) : null}
            </div>
          ) : (
            <div className="empty-box small">
              {predictResult
                ? 'Suggestions hidden because a None/Healthy top-5 result reached 80% confidence.'
                : 'Run image prediction to unlock symptom suggestions.'}
            </div>
          )}

          <div className="search-row">
            <label htmlFor="symptom-search">Search symptom catalog</label>
            <input
              id="symptom-search"
              value={symptomQuery}
              onChange={(event) => setSymptomQuery(event.target.value)}
              placeholder="Example: itching, redness, scaling"
            />
          </div>

          <div className="chip-grid catalog">
            {(catalogLoading ? [] : filteredCatalog).map((item) => {
              const active = selectedSymptoms.includes(item.name)
              return (
                <button
                  key={item.name}
                  type="button"
                  className={`chip subtle ${active ? 'active' : ''}`}
                  onClick={() => toggleSymptom(item.name)}
                >
                  {item.label}
                </button>
              )
            })}
          </div>

          <div className="keyword-block">
            <h3>Custom Symptom Keywords</h3>
            <div className="keyword-input-row">
              <input
                value={keywordInput}
                onChange={(event) => setKeywordInput(event.target.value)}
                placeholder="Type custom symptoms (comma separated)"
              />
              <button type="button" className="btn secondary" onClick={() => addKeywordTerms(keywordInput)}>
                Add
              </button>
            </div>

            {effectiveKeywords.length > 0 ? (
              <div className="chip-grid">
                {effectiveKeywords.map((term) => (
                  <button key={term} type="button" className="chip active" onClick={() => removeKeyword(term)}>
                    {term}
                  </button>
                ))}
              </div>
            ) : (
              <p className="hint-text">No custom keywords added yet.</p>
            )}
          </div>

          {selectedSymptoms.length > 0 ? (
            <div className="selected-wrap">
              <h3>Selected Symptoms</h3>
              <div className="chip-grid">
                {selectedSymptoms.map((name) => (
                  <button key={name} type="button" className="chip active" onClick={() => toggleSymptom(name)}>
                    {symptomLabelMap.get(name) || toLabel(name)}
                  </button>
                ))}
              </div>
            </div>
          ) : null}

          <button
            type="button"
            className="btn primary"
            onClick={handleFinalPredict}
            disabled={finalizing || (selectedSymptoms.length === 0 && effectiveKeywords.length === 0)}
          >
            {finalizing ? 'Calculating Final Prediction...' : 'Finalize Prediction'}
          </button>

          {finalError ? <p className="error-text">{finalError}</p> : null}
        </section>

        <section className="panel final-card">
          <h2>Final Diagnosis</h2>
          {!finalResult ? (
            <div className="empty-box">Select symptoms and/or keywords, then click Finalize Prediction.</div>
          ) : (
            <>
              <button
                type="button"
                className="btn secondary download-report-btn"
                onClick={handleDownloadReport}
                disabled={downloadingReport}
              >
                <span aria-hidden="true">↓</span>
                {downloadingReport ? 'Preparing Report...' : 'Download PDF Report'}
              </button>

              <div className="final-main">
                <p className="final-name">{toLabel(finalResult.final_prediction?.diagnosis)}</p>
                <p>
                  Confidence: <strong>{formatPercent(finalResult.final_prediction?.confidence || 0)}</strong>
                </p>
                <p>
                  Severity: <strong>{toLabel(finalResult.final_prediction?.severity_level || 'Unknown')}</strong>
                </p>
                <p className="recommendation">{finalResult.final_prediction?.treatment_recommendation}</p>
              </div>

              <h3>Differential Top-5</h3>
              <ul className="score-list compact">
                {(finalResult.top_diagnoses || []).slice(0, 5).map((row) => {
                  const width = `${(confidenceToFraction(row.confidence) * 100).toFixed(2)}%`
                  return (
                    <li key={row.diagnosis}>
                      <div className="score-head">
                        <span>{toLabel(row.diagnosis)}</span>
                        <span>{formatPercent(row.confidence)}</span>
                      </div>
                      <div className="bar-track">
                        <div className="bar-fill secondary" style={{ width }} />
                      </div>
                    </li>
                  )
                })}
              </ul>

              <p className="notice">{finalResult.medical_notice}</p>
            </>
          )}
        </section>
      </main>

      <footer className="footer-note">Decision support only. Consult a qualified clinician for diagnosis and care.</footer>
    </div>
  )
}
