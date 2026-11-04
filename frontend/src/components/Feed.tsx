import { useRef, useState, type CSSProperties } from "react"
import { ArrowRight, Play, TriangleAlert, Undo2 } from "lucide-react"
import { Switch } from "@/components/ui/switch"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Mascot } from "./common"
import { maskRtsp, type Box, type Hello, type LiveState, type Status } from "@/lib/spottr"

const pos = (b: Box, g = 0): CSSProperties => ({
  left: `calc(${b[0] * 100}% - ${g}px)`, top: `calc(${b[1] * 100}% - ${g}px)`,
  width: `calc(${(b[2] - b[0]) * 100}% + ${2 * g}px)`, height: `calc(${(b[3] - b[1]) * 100}% + ${2 * g}px)`,
})
const tagAt = (b: Box, g = 0): CSSProperties => ({ left: `calc(${b[0] * 100}% - ${g}px)`, bottom: `calc(${(1 - b[1]) * 100}% + ${g + 3}px)` })
// the same dark glass as the dashboard panels, a little darker so text reads over the static
const PANEL = "rounded-2xl bg-black/60 backdrop-blur-xl border border-white/10 shadow-[inset_0_1px_0_rgba(255,255,255,0.08)]"
const polyPoints = (z: [number, number][]) => z.map(([x, y]) => `${x * 100},${y * 100}`).join(" ")

export interface SetupProps {
  source: string
  setSource: (s: string) => void
  onStart: () => void
  onRecalibrate: () => void
  calibrating: boolean
  calib: string | null
  onSaveZone: (pts: [number, number][]) => void
  onCancelCalib: () => void
}

interface Props extends SetupProps {
  connected: boolean
  hello: Hello | null
  status: Status | null
  live: LiveState | null
  frame: string | null
  selected: number | null
  onSelect: (id: number | null) => void
  showOutline: boolean
  setShowOutline: (v: boolean) => void
  showUnv: boolean
  setShowUnv: (v: boolean) => void
  onCancel: () => void
  onBack: () => void
}

export function Feed(p: Props) {
  const phase = p.status?.phase ?? "idle"
  const view = !p.connected ? "offline" : p.calibrating ? "calibrate"
    : phase === "connecting" ? "connecting" : phase === "live" ? "live" : phase === "error" ? "error" : "setup"
  return (
    // live: a 16:9 picture, centred and sized to fit the screen.
    // setup / connecting / errors: no picture yet, so use the whole column (the panels fit on small screens)
    <div className={`relative rounded-2xl overflow-hidden bg-black border border-white/10 w-full mx-auto ${view === "live"
      ? "aspect-video max-w-[calc((100vh-330px)*16/9)] shrink-0" : "flex-1 min-h-[440px]"} ${p.live && p.live.top !== "OK" && p.live.top !== "WATCH" ? "glow-red" : ""}`}>
      {view === "live" && p.frame && <img src={p.frame} className="absolute inset-0 size-full object-cover" alt="" />}
      {view === "live" && p.live && <LiveOverlay {...p} />}
      {view === "live" && p.status?.frozen && <Frozen status={p.status} />}
      {view === "setup" && <Setup {...p} />}
      {view === "calibrate" && <Calibrate {...p} />}
      {view === "connecting" && <Connecting status={p.status!} hello={p.hello} onCancel={p.onCancel} />}
      {view === "error" && <ErrorCard message={p.status?.message ?? ""} onBack={p.onBack} />}
      {view === "offline" && <Offline />}
    </div>
  )
}

function LiveOverlay({ live, hello, selected, onSelect, showOutline, setShowOutline, showUnv, setShowUnv }: Props) {
  const L = live!
  return (
    <div className="absolute inset-0" onClick={() => onSelect(null)}>
      {showOutline && hello?.zone && (
        <svg viewBox="0 0 100 100" preserveAspectRatio="none" className="absolute inset-0 size-full pointer-events-none">
          <polygon points={polyPoints(hello.zone)} fill="rgba(8,102,255,.07)" stroke="#5aa0ff" strokeWidth="2" strokeDasharray="7 5" vectorEffect="non-scaling-stroke" />
        </svg>
      )}
      {showUnv && L.unverified.map((u) => (
        <div key={u.id}>
          <div className="absolute border-2 border-dashed border-neutral-300/70 rounded" style={pos(u.box)} />
          <div className="absolute text-[11px] font-semibold bg-neutral-800/90 text-neutral-300 rounded px-1 whitespace-nowrap" style={tagAt(u.box)}>? unverified</div>
        </div>
      ))}
      {L.deck.map((d) => (
        <div key={d.id} onClick={(e) => { e.stopPropagation(); onSelect(d.id) }} className="cursor-pointer">
          <div className="absolute border-2 border-neutral-300/80 border-dashed rounded" style={pos(d.box)} />
          {selected === d.id && <div className="absolute rounded-md border-[3px] border-white glow-white" style={pos(d.box, 6)} />}
          <div className="absolute text-[11px] font-semibold bg-neutral-700/90 text-white rounded px-1 whitespace-nowrap" style={tagAt(d.box, selected === d.id ? 6 : 0)}>ID {d.id} deck</div>
        </div>
      ))}
      {L.swimmers.map((w) => {
        const sel = selected === w.id
        const click = (e: React.MouseEvent) => { e.stopPropagation(); onSelect(w.id) }
        if (w.state === "OUTVIEW")
          return (
            <div key={w.id} onClick={click} className="absolute cursor-pointer text-[12px] font-bold bg-white/90 text-black rounded px-1.5 whitespace-nowrap inline-flex items-center gap-1"
              style={{ top: `${w.box[1] * 100}%`, ...(w.side === "right" ? { right: "0.6%" } : { left: "0.6%" }), outline: sel ? "3px solid #fff" : undefined }}>
              {w.side === "left" && <ArrowRight className="size-[12px] rotate-180" />}ID {w.id}{w.side === "right" && <ArrowRight className="size-[12px]" />} out of view
            </div>
          )
        if (w.state === "LOST")
          return (
            <div key={w.id} onClick={click} className="cursor-pointer">
              <div className="absolute rounded-full border-2 border-dashed border-red-400" style={pos(w.box, 14)} />
              <div className="absolute text-[11px] font-bold bg-red-900/90 text-red-100 rounded px-1.5 whitespace-nowrap" style={tagAt(w.box, 14)}>ID {w.id} last seen</div>
            </div>
          )
        const lvl = w.alarm
        const red = lvl === "ALARM" || lvl === "EMERGENCY"
        const g = lvl === "OK" ? 0 : 10 + w.border
        return (
          <div key={w.id} onClick={click} className="cursor-pointer">
            {lvl === "OK"
              ? <div className="absolute border-2 border-muse rounded" style={pos(w.box)} />
              : <div className={`absolute rounded-lg ${red ? "animate-flash glow-red" : ""} ${lvl === "EMERGENCY" ? "border-red-700" : red ? "border-red-600" : "border-amber-400"}`}
                  style={{ ...pos(w.box, g), borderWidth: Math.max(4, w.border) }} />}
            {sel && <div className="absolute rounded-md border-[3px] border-white glow-white" style={pos(w.box, g + 8)} />}
            <div className={`absolute font-bold rounded px-1.5 py-0.5 whitespace-nowrap ${lvl === "OK" ? "text-[12px] bg-muse text-white" : red ? `text-[13px] text-white ${lvl === "EMERGENCY" ? "bg-red-800" : "bg-red-600"}` : "text-[13px] bg-amber-400 text-neutral-900"}`}
              style={tagAt(w.box, sel ? g + 8 : g)}>
              ID {w.id}{lvl === "OK" ? "" : ` · ${lvl === "WATCH" ? "" : lvl + " · "}${w.reason.length > 34 ? w.reason.slice(0, 32) + "…" : w.reason}`}
            </div>
          </div>
        )
      })}
      <div className="absolute left-3 bottom-3 flex gap-2 text-[12px]" onClick={(e) => e.stopPropagation()}>
        <label className="rounded-lg bg-black/70 backdrop-blur text-neutral-200 px-2.5 py-1.5 flex items-center gap-2 cursor-pointer">
          <Switch checked={showOutline} onCheckedChange={setShowOutline} />Pool outline
        </label>
        <label className="rounded-lg bg-black/70 backdrop-blur text-neutral-200 px-2.5 py-1.5 flex items-center gap-2 cursor-pointer">
          <Switch checked={showUnv} onCheckedChange={setShowUnv} />Show unverified
        </label>
      </div>
    </div>
  )
}

function Steps({ at }: { at: 1 | 2 | 3 }) {
  const s = (n: number, label: string) => (
    <span className={`rounded-full px-2.5 py-1 ${n === at ? "bg-muse text-white" : n < at ? "border border-neutral-700 text-neutral-400" : "border border-neutral-700 text-neutral-300"}`}>
      {n} {label}{n < at ? " ✓" : ""}
    </span>
  )
  return (
    <div className="flex items-center gap-2 text-[12px] font-semibold">
      {s(1, "Camera")}<span className="text-neutral-600">—</span>{s(2, "Pool outline")}<span className="text-neutral-600">—</span>{s(3, "Start")}
    </div>
  )
}

function Setup({ source, setSource, onStart, onRecalibrate, hello, status }: Props) {
  const [focus, setFocus] = useState(false)
  const stopped = status?.phase === "stopped"
  return (
    <div className="absolute inset-0 tv-static flex p-5 sm:p-8 overflow-y-auto">
      <div className={`${PANEL} m-auto w-[600px] max-w-full p-5`}>
        <Steps at={1} />
        {stopped && status?.message && <div className="mt-3 rounded-lg bg-white/5 border border-white/10 px-3 py-2 text-[13px] text-neutral-300">{status.message}{status.log ? ` Log saved: ${status.log}` : ""}</div>}
        <div className="text-[20px] font-bold mt-3">Connect the pool camera</div>
        <label className="block text-[12px] font-semibold text-neutral-400 mt-3 mb-1.5">RTSP address</label>
        <div className="relative">
          <input value={source} onChange={(e) => setSource(e.target.value)} onFocus={() => setFocus(true)} onBlur={() => setFocus(false)}
            placeholder="rtsp://user:password@192.168.137.179:554/stream1" spellCheck={false}
            className={`w-full rounded-lg bg-white/[0.06] border border-white/15 px-3 py-2.5 font-mono text-[14px] outline-none focus:border-muse ${focus ? "text-neutral-200" : "text-transparent caret-transparent"}`} />
          {!focus && <div className="pointer-events-none absolute inset-0 px-3 py-2.5 font-mono text-[14px] text-neutral-200 truncate">{source ? maskRtsp(source) : <span className="text-neutral-600">rtsp://user:password@192.168.137.179:554/stream1</span>}</div>}
        </div>
        <div className="flex items-center gap-3 mt-2 text-[12px] text-neutral-500">
          <span>remembered on this laptop · or play a recorded clip</span>
          <Select value={hello?.clips.includes(source) ? source : ""} onValueChange={setSource}>
            <SelectTrigger size="sm" className="h-7 max-w-[220px] bg-white/[0.06] border-white/15 text-neutral-300 text-[12px] hover:bg-white/10">
              <SelectValue placeholder="Choose clip…" />
            </SelectTrigger>
            <SelectContent className="bg-neutral-900/95 backdrop-blur-xl border-white/15 text-neutral-300">
              {hello?.clips.map((c) => (
                <SelectItem key={c} value={c} className="text-[12.5px] text-neutral-300 focus:bg-white/10 focus:text-white">{c.split("/").pop()}</SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>
        <div className="mt-3 flex items-center gap-4 rounded-xl border border-white/10 bg-white/[0.04] p-3">
          <div className="relative w-[150px] aspect-video rounded-md overflow-hidden bg-neutral-800 shrink-0">
            {hello?.zone_thumb && <img src={`data:image/jpeg;base64,${hello.zone_thumb}`} className="size-full object-cover opacity-80" alt="" />}
            {hello?.zone && (
              <svg viewBox="0 0 100 100" preserveAspectRatio="none" className="absolute inset-0 size-full">
                <polygon points={polyPoints(hello.zone)} fill="rgba(8,102,255,.25)" stroke="#5aa0ff" strokeWidth="1.2" vectorEffect="non-scaling-stroke" />
              </svg>
            )}
          </div>
          <div className="text-[13px]">
            {hello?.zone ? <>
              <div className="font-semibold text-neutral-100">Pool outline saved</div>
              <div className="text-neutral-400">{hello.zone.length} corners{hello.zone_saved ? ` · ${hello.zone_saved}` : ""}</div>
            </> : <>
              <div className="font-semibold text-amber-300">No pool outline yet</div>
              <div className="text-neutral-400">Click the water's corners once per camera position.</div>
            </>}
            <button onClick={onRecalibrate} disabled={!source}
              className="mt-2 rounded-md border border-white/20 bg-white/[0.06] px-2 py-0.5 text-neutral-200 font-medium hover:border-white/50 disabled:opacity-40">
              {hello?.zone ? "Recalibrate" : "Draw outline"}
            </button>
          </div>
        </div>
        <button onClick={onStart} disabled={!source}
          className="mt-4 w-full rounded-xl bg-muse text-white font-bold text-[16px] py-3 flex items-center justify-center gap-2 glow-blue disabled:opacity-40 hover:brightness-110">
          <Play className="size-[18px] fill-current" />Start monitoring
        </button>
        <div className="text-[12px] text-neutral-500 mt-2.5 text-center">Starts the GPUs (first start ~1–2 min) and turns on the alarm sound.</div>
      </div>
    </div>
  )
}

function Calibrate({ calib, onSaveZone, onCancelCalib }: Props) {
  const [pts, setPts] = useState<[number, number][]>([])
  const ref = useRef<HTMLDivElement>(null)
  const add = (e: React.MouseEvent) => {
    const r = ref.current!.getBoundingClientRect()
    setPts((p) => [...p, [+((e.clientX - r.left) / r.width).toFixed(4), +((e.clientY - r.top) / r.height).toFixed(4)]])
  }
  return (
    <div className="absolute inset-0">
      {!calib ? (
        <div className="absolute inset-0 tv-static flex items-center justify-center text-neutral-200 font-semibold">Getting a picture from the camera…</div>
      ) : (
        <div ref={ref} className="absolute inset-0 cursor-crosshair" onClick={add}>
          <img src={`data:image/jpeg;base64,${calib}`} className="size-full object-cover" alt="" draggable={false} />
          <svg viewBox="0 0 100 100" preserveAspectRatio="none" className="absolute inset-0 size-full pointer-events-none">
            {pts.length >= 3 && <polygon points={polyPoints(pts)} fill="rgba(8,102,255,.15)" stroke="none" />}
            <polyline points={polyPoints(pts)} fill="none" stroke="#0866FF" strokeWidth="3" vectorEffect="non-scaling-stroke" />
          </svg>
          {pts.map(([x, y], i) => (
            <div key={i} className="absolute size-3.5 -ml-[7px] -mt-[7px] rounded-full bg-white border-[3px] border-muse pointer-events-none" style={{ left: `${x * 100}%`, top: `${y * 100}%` }} />
          ))}
        </div>
      )}
      <div className={`${PANEL} absolute right-4 bottom-4 w-[430px] max-w-[calc(100%-2rem)] p-4`} onClick={(e) => e.stopPropagation()}>
        <Steps at={2} />
        <div className="text-[18px] font-bold mt-3">Click the corners of the WATER</div>
        <div className="text-[13px] text-neutral-400 mt-1">Go around the edge of the water (not the deck). People outside it count as "on deck".</div>
        <div className="flex items-center gap-2 mt-3 text-[13px]">
          <span className="font-mono text-neutral-300">{pts.length} corners</span>
          <button onClick={() => setPts((p) => p.slice(0, -1))} className="ml-auto rounded-md border border-neutral-600 px-2.5 py-1 inline-flex items-center gap-1"><Undo2 className="size-[13px]" />Undo</button>
          <button onClick={onCancelCalib} className="rounded-md border border-neutral-600 px-2.5 py-1">Cancel</button>
          <button onClick={() => onSaveZone(pts)} disabled={pts.length < 3} className="rounded-md bg-muse text-white font-semibold px-3 py-1 disabled:opacity-40">Save outline</button>
        </div>
      </div>
    </div>
  )
}

function Connecting({ status, hello, onCancel }: { status: Status; hello: Hello | null; onCancel: () => void }) {
  const live = status.gpus.filter((g) => g.state === "live").length
  const total = status.gpus_total || 1
  const row = (label: string, state: string, cls: string, dot: string) => (
    <div className="flex items-center gap-2"><span className={`size-2.5 rounded-full ${dot}`} />{label}<span className={`ml-auto ${cls}`}>{state}</span></div>
  )
  const ok = "text-muse-soft font-semibold"
  return (
    <div className="absolute inset-0 tv-static flex p-5 sm:p-8 overflow-y-auto">
      <div className={`${PANEL} m-auto w-[600px] max-w-full p-5`}>
        <div className="flex items-center gap-4">
          <Mascot className="h-[78px] w-auto" />
          <div>
            <div className="text-[20px] font-bold">Spottr is getting ready…</div>
            <div className="text-[13px] text-neutral-400 mt-1">{status.message || "GPUs start in parallel. The first start takes 1–2 minutes."}</div>
          </div>
        </div>
        <div className="mt-4 h-2 rounded-full bg-white/10 overflow-hidden"><div className="h-full bg-muse" style={{ width: `${Math.max(6, (live / total) * 100)}%` }} /></div>
        <div className="mt-4 grid grid-cols-2 gap-x-6 gap-y-2 text-[14px]">
          {status.camera === "connected" ? row("Camera", "connected", ok, "bg-muse") : row("Camera", "connecting…", "text-amber-300", "bg-amber-400 animate-flash")}
          {hello?.zone ? row("Pool outline", "loaded", ok, "bg-muse") : row("Pool outline", "none — whole frame", "text-amber-300", "bg-amber-400")}
          {status.gpus.map((g) => (
            <div key={g.i}>{g.state === "live" ? row(`GPU ${g.i}`, "live", ok, "bg-muse")
              : g.state === "error" ? row(`GPU ${g.i}`, g.msg || "failed", "text-red-300 truncate max-w-[150px]", "bg-red-500")
              : row(`GPU ${g.i}`, "starting · loading SAM 3.1…", "text-amber-300", "bg-amber-400 animate-flash")}</div>
          ))}
          {status.muse ? row("Muse Spark", "key found", ok, "bg-muse") : row("Muse Spark", "off (no key)", "text-neutral-400", "bg-neutral-600")}
        </div>
        <div className="mt-5 flex items-center gap-3 text-[13px]">
          <span className="text-neutral-400">Monitoring starts as soon as 3 GPUs are live.</span>
          <button onClick={onCancel} className="ml-auto rounded-md border border-neutral-600 px-3 py-1.5 hover:border-white">Cancel</button>
        </div>
      </div>
    </div>
  )
}

function Frozen({ status }: { status: Status }) {
  return (
    <div className="absolute inset-0 bg-black/55 backdrop-grayscale flex items-center justify-center">
      <div className="w-[600px] max-w-[92%] rounded-2xl bg-red-700 text-white p-6 glow-red">
        <div className="flex items-center gap-3"><TriangleAlert className="size-[30px]" />
          <div className="text-[24px] font-extrabold">FEED FROZEN — watch the pool yourself</div></div>
        <div className="text-[15px] text-red-100 mt-2">
          The camera picture hasn't changed for <b>{Math.round(status.frozen_s ?? 4)} s</b>. Spottr can't see the pool, so no alarms can fire.
          {status.camera === "reconnecting" ? ` Reconnecting to the camera… (attempt ${status.reconnects ?? 1})` : ""}
        </div>
        <div className="flex gap-2 mt-4 text-[13px]">
          <span className="rounded-md bg-black/30 px-2 py-1">GPU streams {status.gpus_live} / {status.gpus_total} live</span>
        </div>
      </div>
    </div>
  )
}

function ErrorCard({ message, onBack }: { message: string; onBack: () => void }) {
  return (
    <div className="absolute inset-0 tv-static flex p-5 sm:p-8 overflow-y-auto">
      <div className={`${PANEL} m-auto w-[560px] max-w-full p-5 !border-red-500/60`}>
        <div className="flex items-center gap-3 text-red-300"><TriangleAlert className="size-[26px]" /><div className="text-[20px] font-bold text-white">Couldn't start monitoring</div></div>
        <div className="text-[14px] text-neutral-300 mt-2">{message}</div>
        <button onClick={onBack} className="mt-4 rounded-lg bg-white/10 border border-white/15 px-4 py-2 font-semibold hover:bg-white/15">Back to setup</button>
      </div>
    </div>
  )
}

function Offline() {
  return (
    <div className="absolute inset-0 tv-static flex p-5 sm:p-8 overflow-y-auto">
      <div className={`${PANEL} m-auto w-[560px] max-w-full p-5 flex gap-4 items-center`}>
        <Mascot className="h-[90px] w-auto" />
        <div>
          <div className="text-[20px] font-bold">Spottr isn't running on this laptop</div>
          <div className="text-[14px] text-neutral-400 mt-1">Start it in a terminal: <span className="font-mono text-neutral-200">python spottr_app.py</span>. This page connects by itself.</div>
        </div>
      </div>
    </div>
  )
}
