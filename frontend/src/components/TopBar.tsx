import { Cpu, Maximize, Sparkles, TriangleAlert, Volume2, VolumeX } from "lucide-react"
import { Mascot } from "./common"
import type { LiveState, Status, Swimmer } from "@/lib/spottr"

interface Props {
  status: Status | null
  live: LiveState | null
  muse: boolean
  muted: boolean
  onMute: () => void
  onAckAll: () => void
  worst: Swimmer | null
  clock: string
}

const Sep = () => <span className="mx-2 text-neutral-700">|</span>

export function TopBar({ status, live, muse, muted, onMute, onAckAll, worst, clock }: Props) {
  const phase = status?.phase ?? "idle"
  const alarm = live?.top === "ALARM" && worst
  const more = live ? live.swimmers.filter((w) => w.alarm === "ALARM" || w.alarm === "EMERGENCY").length - 1 : 0
  return (
    <header className="relative z-10 h-[76px] shrink-0 px-6 flex items-center gap-4 bg-black/40 backdrop-blur-xl border-b border-white/10">
      <Mascot className="h-[60px] w-auto -my-2 select-none cursor-pointer" />
      <div>
        <div className="text-[26px] font-extrabold tracking-tight leading-none">Spottr</div>
        <div className="text-[12px] text-neutral-500 font-medium mt-1">AI Lifeguard Assistant</div>
      </div>

      {alarm ? (
        <div className="ml-8 flex-1 flex items-center gap-3 rounded-xl bg-red-600 text-white px-4 py-2.5 glow-red animate-flash">
          <TriangleAlert className="size-[22px]" />
          <div className="font-bold text-[17px]">ALARM — ID {worst.id} {worst.reason}</div>
          {more > 0 && <div className="text-red-100 text-[14px]">+{more} more alarm{more > 1 ? "s" : ""}</div>}
          <button onClick={onAckAll} className="ml-auto rounded-lg bg-white text-red-700 font-semibold text-[14px] px-4 py-1.5">
            Acknowledge all <span className="font-mono text-[12px] text-red-400 ml-1">A</span>
          </button>
        </div>
      ) : live?.top === "EMERGENCY" ? null : (
        <div className="ml-auto flex items-center gap-2 text-[13px]">
          {phase === "live" && !status?.frozen && (
            <span className="inline-flex items-center gap-2 rounded-full bg-muse/15 text-muse-soft border border-muse/40 px-3 py-1 font-semibold">
              <span className="size-2 rounded-full bg-muse" />LIVE · {status?.source}
            </span>
          )}
          {phase === "live" && status?.frozen && (
            <span className="inline-flex items-center gap-2 rounded-full bg-red-600/20 text-red-300 border border-red-500/50 px-3 py-1 font-semibold">
              <span className="size-2 rounded-full bg-red-500 animate-flash" />FEED FROZEN
            </span>
          )}
          {phase === "connecting" && (
            <span className="inline-flex items-center gap-2 rounded-full glass px-3 py-1 font-semibold text-amber-300">
              <span className="size-2 rounded-full bg-amber-400 animate-flash" />
              Starting · {status?.gpus_live ?? 0} of {status?.gpus_total ?? 0} GPUs live
            </span>
          )}
          {(phase === "idle" || phase === "stopped" || phase === "error") && (
            <span className="inline-flex items-center gap-2 rounded-full glass px-3 py-1 font-semibold text-neutral-400">
              <span className="size-2 rounded-full bg-neutral-500" />Not monitoring
            </span>
          )}
          {live && (
            <span className="inline-flex items-center rounded-full glass px-3 py-1 text-neutral-400">
              <Cpu className="size-[14px] text-neutral-300 mr-1.5" />
              <b className="font-mono text-white">{live.perf.checks_s.toFixed(1)}</b>&nbsp;checks/s<Sep />
              delay&nbsp;<b className="font-mono text-white">{(live.perf.delay_ms / 1000).toFixed(2)} s</b><Sep />
              GPU&nbsp;<b className="font-mono text-white">{live.perf.gpu_ms} ms</b><Sep />
              <b className="font-mono text-white">{live.perf.gpus_live}/{live.perf.gpus_total}</b>&nbsp;GPUs<Sep />
              <b className="font-mono text-white">{live.perf.errors}</b>&nbsp;errors
            </span>
          )}
          <span className={`inline-flex items-center gap-1.5 rounded-full px-3 py-1 font-semibold ${muse ? "bg-muse text-white" : "glass text-neutral-500"}`}>
            <Sparkles className="size-[13px]" />Muse Spark{muse ? "" : " · off"}
          </span>
          <button onClick={onMute} className="inline-flex items-center gap-1.5 rounded-full glass px-3 py-1 text-neutral-300 hover:text-white">
            {muted ? <VolumeX className="size-[15px]" /> : <Volume2 className="size-[15px]" />}{muted ? "Muted" : "Sound"}
          </button>
          {live && (
            <button onClick={onAckAll} className="rounded-full glass px-3 py-1 text-neutral-200 font-semibold hover:text-white">
              Ack all <span className="font-mono text-neutral-500 text-[11px] ml-1">A</span>
            </button>
          )}
          <button
            title="Full screen (F)"
            onClick={() => (document.fullscreenElement ? document.exitFullscreen() : document.documentElement.requestFullscreen())}
            className="inline-flex items-center justify-center rounded-full glass w-[30px] h-[28px] text-neutral-200 hover:text-white"
          >
            <Maximize className="size-[15px]" />
          </button>
          <span className="font-mono text-neutral-500 ml-1">{clock}</span>
        </div>
      )}
    </header>
  )
}

export function EmergencyBanner({ worst, onAckAll }: { worst: Swimmer; onAckAll: () => void }) {
  return (
    <div className="relative z-40 h-[64px] shrink-0 flex items-stretch">
      <div className="hazard w-[18px]" />
      <div className="flex-1 flex items-center gap-4 bg-red-700 text-white px-5 animate-flash">
        <TriangleAlert className="size-[30px]" />
        <div className="text-[24px] font-extrabold tracking-tight uppercase">EMERGENCY — ID {worst.id} {worst.reason}</div>
        <div className="text-[16px] font-semibold text-red-100">GO NOW</div>
        <button onClick={onAckAll} className="ml-auto rounded-lg bg-white text-red-700 font-extrabold text-[15px] px-5 py-2">
          Acknowledge all <span className="font-mono text-[12px] text-red-400 ml-1">A</span>
        </button>
      </div>
      <div className="hazard w-[18px]" />
    </div>
  )
}
