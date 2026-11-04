// Connection to spottr_app.py (the laptop backend) + the message types it sends.
import { useCallback, useEffect, useRef, useState } from "react"

export type Level = "OK" | "WATCH" | "ALARM" | "EMERGENCY"
export type Box = [number, number, number, number]

export interface Muse {
  asked_t: number
  asked_under?: number
  pending?: boolean
  answer_s?: number
  verdict?: "ok" | "concern" | "emergency"
  reason?: string
  confidence?: number
}

export interface Swimmer {
  id: number
  box: Box
  state: "SURFACE" | "SUBMERGED" | "UNSEEN" | "LOST" | "OUTVIEW"
  under_s: number
  still_s: number
  head_known: boolean
  head_up: boolean
  alarm: Level
  reason: string
  alarm_since: number
  acked: boolean
  border: number
  next: { level: Level; in_s: number } | null
  muse: Muse | null
  lost_at_edge: boolean
  side: "left" | "right"
  seen_ago: number
  in_pool_s: number
}

export interface SpottrEvent { t: number; wall: string; kind: string; id: number | null; text: string }

export interface LiveState {
  t: number
  session_s: number
  perf: { checks_s: number; delay_ms: number; gpu_ms: number; gpus_live: number; gpus_total: number; errors: number }
  counts: { pool: number; deck: number; unverified: number; under: number; outview: number; lost: number; above: number; entries: number; exits: number }
  swimmers: Swimmer[]
  deck: { id: number; box: Box }[]
  unverified: { id: number; box: Box }[]
  top: Level
  sounding: boolean
  events: SpottrEvent[]
}

export interface Status {
  phase: "idle" | "connecting" | "live" | "error" | "stopped"
  camera: string
  gpus: { i: number; state: "starting" | "live" | "dropped" | "error"; msg: string }[]
  gpus_live: number
  gpus_total: number
  frozen: boolean
  frozen_s?: number
  reconnects?: number
  muse: boolean
  source?: string
  message: string
  log?: string | null
}

export interface Hello {
  zone: [number, number][] | null
  zone_saved: string | null
  zone_thumb: string | null
  clips: string[]
  muse: boolean
  streams: number
  thresholds: Record<string, number>
}

const WS_URL = `ws://${location.hostname === "localhost" ? "127.0.0.1" : location.hostname || "127.0.0.1"}:8765/ws`

export function useSpottr() {
  const [connected, setConnected] = useState(false)
  const [hello, setHello] = useState<Hello | null>(null)
  const [status, setStatus] = useState<Status | null>(null)
  const [live, setLive] = useState<LiveState | null>(null)
  const [frame, setFrame] = useState<string | null>(null)
  const [calib, setCalib] = useState<string | null>(null)
  const [report, setReport] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)
  const ws = useRef<WebSocket | null>(null)
  const lastUrl = useRef<string | null>(null)

  useEffect(() => {
    let stop = false
    let timer: number | undefined
    const open = () => {
      const s = new WebSocket(WS_URL)
      s.binaryType = "blob"
      ws.current = s
      s.onopen = () => setConnected(true)
      s.onclose = () => {
        setConnected(false)
        if (!stop) timer = window.setTimeout(open, 1500) // keep trying: the backend may start later
      }
      s.onmessage = (e) => {
        if (e.data instanceof Blob) {
          const url = URL.createObjectURL(e.data)
          if (lastUrl.current) URL.revokeObjectURL(lastUrl.current)
          lastUrl.current = url
          setFrame(url)
          return
        }
        const m = JSON.parse(e.data)
        if (m.type === "hello") setHello(m)
        else if (m.type === "status") {
          setStatus(m)
          if (m.phase !== "live") { setLive(null); setFrame(null) }
        } else if (m.type === "state") setLive(m)
        else if (m.type === "calib_frame") setCalib(m.jpeg)
        else if (m.type === "report") setReport(m.text)
        else if (m.type === "error") setError(m.message)
      }
    }
    open()
    return () => { stop = true; window.clearTimeout(timer); ws.current?.close() }
  }, [])

  const send = useCallback((m: object) => {
    if (ws.current?.readyState === WebSocket.OPEN) ws.current.send(JSON.stringify(m))
  }, [])

  return { connected, hello, status, live, frame, calib, setCalib, report, setReport, error, setError, send }
}

// ---- alarm sound (Web Audio). Browsers only allow sound after a click, so
// the AudioContext is created by the "Start monitoring" click.
let ctx: AudioContext | null = null
export function unlockSound() {
  if (!ctx) ctx = new AudioContext()
  if (ctx.state === "suspended") void ctx.resume()
}
function beep(freq: number, ms: number) {
  if (!ctx) return
  const o = ctx.createOscillator()
  const g = ctx.createGain()
  o.type = "square"
  o.frequency.value = freq
  g.gain.value = 0.08
  o.connect(g).connect(ctx.destination)
  o.start()
  o.stop(ctx.currentTime + ms / 1000)
}
export function useAlarmSound(on: boolean, level: Level) {
  useEffect(() => {
    if (!on) return
    const fast = level === "EMERGENCY"
    beep(fast ? 1400 : 1000, 220)
    const id = window.setInterval(() => beep(fast ? 1400 : 1000, 220), fast ? 350 : 650)
    return () => window.clearInterval(id)
  }, [on, level])
}

export const fmtClock = (s: number) => {
  const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60), x = Math.floor(s % 60)
  return [h, m, x].map((v) => String(v).padStart(2, "0")).join(":")
}

export function maskRtsp(url: string) {
  return url.replace(/(rtsp:\/\/[^:/@]+:)([^@]+)(@)/, (_, a, b, c) => a + "•".repeat(Math.min(b.length, 10)) + c)
}
