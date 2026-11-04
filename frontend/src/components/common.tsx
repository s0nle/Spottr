import { useEffect, useState, type ComponentType } from "react"
import {
  BadgeCheck, Cpu, EyeOff, LogIn, LogOut, Play, Search, Sparkles, Timer, TriangleAlert,
  UserPlus, VideoOff, Waves, CircleCheck, CircleSlash,
} from "lucide-react"
import type { Level } from "@/lib/spottr"

export const LEVEL_BADGE: Record<Level, string> = {
  OK: "bg-muse text-white",
  WATCH: "bg-amber-400 text-neutral-900",
  ALARM: "bg-red-600 text-white",
  EMERGENCY: "bg-red-900 text-white ring-1 ring-red-400",
}

export const LEVEL_RING: Record<Level, string> = {
  OK: "border-muse",
  WATCH: "border-amber-400",
  ALARM: "border-red-600",
  EMERGENCY: "border-red-700",
}

// one colour family per event kind (status kinds keep status colours)
export const EVENT_STYLE: Record<string, [string, ComponentType<{ className?: string }>, string?]> = {
  EMERGENCY: ["bg-red-900 text-white ring-1 ring-red-400", TriangleAlert],
  ALARM: ["bg-red-600 text-white", TriangleAlert],
  LOST: ["bg-red-600/25 text-red-200 ring-1 ring-red-500/60", Search],
  DROPPED: ["bg-red-600/25 text-red-200 ring-1 ring-red-500/60", CircleSlash],
  WATCH: ["bg-amber-400 text-neutral-900", Timer],
  MUSE: ["bg-muse text-white", Sparkles],
  FOUND: ["bg-muse/20 text-[#9cc3ff] ring-1 ring-muse/50", Search],
  VERIFIED: ["bg-muse/20 text-[#9cc3ff] ring-1 ring-muse/50", BadgeCheck],
  CLEAR: ["bg-muse/20 text-[#9cc3ff] ring-1 ring-muse/50", CircleCheck, "OK"],
  NEW: ["bg-white/10 text-neutral-200", UserPlus],
  ENTER: ["bg-white/15 text-neutral-100", LogIn],
  EXIT: ["bg-white/15 text-neutral-100", LogOut],
  DIVE: ["bg-white/15 text-neutral-100", Waves],
  OUT: ["bg-white text-black", EyeOff, "OUT OF VIEW"],
  FEED: ["bg-red-700 text-white", VideoOff],
  GPU: ["bg-white/15 text-neutral-100", Cpu],
  START: ["bg-white/15 text-neutral-100", Play],
}

// The lifeguard mascot. Hover over it and it runs: 3 frames (the logo + two
// running strides made with Muse Image) at ~9 frames/s with a small bounce.
const RUN = ["/run0.png", "/run1.png", "/run2.png"]
RUN.forEach((src) => { const i = new Image(); i.src = src })  // preload: no flicker on first hover

export function Mascot({ className = "" }: { className?: string }) {
  const [running, setRunning] = useState(false)
  const [f, setF] = useState(0)
  useEffect(() => {
    if (!running) { setF(0); return }
    const id = window.setInterval(() => setF((x) => (x + 1) % RUN.length), 110)
    return () => window.clearInterval(id)
  }, [running])
  return (
    <img src={RUN[f]} alt="" draggable={false}
      onMouseEnter={() => setRunning(true)} onMouseLeave={() => setRunning(false)}
      className={className}
      style={{ transform: running ? `translateY(${f === 1 ? -4 : f === 2 ? -1 : 0}px) rotate(${f === 1 ? -3 : 0}deg)` : undefined }} />
  )
}
