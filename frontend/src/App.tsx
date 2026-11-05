import { useEffect, useMemo, useState } from "react"
import { X } from "lucide-react"
import { Dialog, DialogContent, DialogHeader, DialogTitle } from "@/components/ui/dialog"
import { TopBar, EmergencyBanner } from "@/components/TopBar"
import { LeftPanel } from "@/components/LeftPanel"
import { Feed } from "@/components/Feed"
import { Events } from "@/components/Events"
import { Swimmers } from "@/components/Swimmers"
import { unlockSound, useAlarmSound, useSpottr, type Swimmer } from "@/lib/spottr"

const RANK = { OK: 0, WATCH: 1, ALARM: 2, EMERGENCY: 3 } as const

function load(key: string, dflt: string) {
  try { return localStorage.getItem(key) ?? dflt } catch { return dflt }
}
function save(key: string, v: string) {
  try { localStorage.setItem(key, v) } catch { /* private window: fine */ }
}

export default function App() {
  const s = useSpottr()
  const [source, setSource] = useState(() => load("spottr.source", ""))
  const [calibrating, setCalibrating] = useState(false)
  const [selected, setSelected] = useState<number | null>(null)
  const [showOutline, setShowOutline] = useState(() => load("spottr.outline", "1") === "1")
  const [showUnv, setShowUnv] = useState(() => load("spottr.unverified", "0") === "1")
  const [muted, setMuted] = useState(false)
  const [reportBusy, setReportBusy] = useState(false)
  const [clock, setClock] = useState(() => new Date().toLocaleTimeString([], { hour12: false }))

  useEffect(() => { const id = setInterval(() => setClock(new Date().toLocaleTimeString([], { hour12: false })), 1000); return () => clearInterval(id) }, [])
  useEffect(() => save("spottr.outline", showOutline ? "1" : "0"), [showOutline])
  useEffect(() => save("spottr.unverified", showUnv ? "1" : "0"), [showUnv])
  useEffect(() => { if (s.report) setReportBusy(false) }, [s.report])

  const live = s.live
  const worst: Swimmer | null = useMemo(() => {
    if (!live) return null
    return [...live.swimmers].sort((a, b) => Number(a.acked) - Number(b.acked) || RANK[b.alarm] - RANK[a.alarm] || b.alarm_since - a.alarm_since)[0] ?? null
  }, [live])
  const top = live?.top ?? "OK"

  useAlarmSound(!!live?.sounding && !muted && (top === "ALARM" || top === "EMERGENCY"), top)

  const ackAll = () => s.send({ cmd: "ack_all" })
  const start = () => { unlockSound(); save("spottr.source", source); s.send({ cmd: "start", source }) }

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if ((e.target as HTMLElement)?.tagName === "INPUT" || (e.target as HTMLElement)?.tagName === "SELECT") return
      const k = e.key.toLowerCase()
      if (k === "a") ackAll()
      else if (k === "m") setMuted((m) => !m)
      else if (k === "f") void (document.fullscreenElement ? document.exitFullscreen() : document.documentElement.requestFullscreen())
      else if (k === "escape") setSelected(null)
    }
    window.addEventListener("keydown", onKey)
    return () => window.removeEventListener("keydown", onKey)
  })

  return (
    <div className="relative h-screen w-screen overflow-hidden bg-black flex flex-col">
      {/* soft Muse-blue light behind the glass */}
      <div className="pointer-events-none absolute inset-0">
        <div className="absolute -left-40 top-40 w-[520px] h-[520px] rounded-full bg-muse/30 blur-[120px]" />
        <div className="absolute right-[-120px] top-[-80px] w-[560px] h-[460px] rounded-full bg-muse/25 blur-[130px]" />
        <div className="absolute left-[38%] bottom-[-220px] w-[700px] h-[420px] rounded-full bg-[#1d4ed8]/25 blur-[140px]" />
      </div>
      {top === "ALARM" && <div className="pointer-events-none absolute inset-0 z-50 border-[14px] border-red-600 animate-flash" />}
      {top === "EMERGENCY" && <div className="pointer-events-none absolute inset-0 z-50 border-[26px] border-red-600 animate-flash" />}

      <TopBar status={s.status} live={live} muse={!!s.hello?.muse} muted={muted} onMute={() => setMuted((m) => !m)}
        onAckAll={ackAll} worst={worst} clock={clock} />
      {top === "EMERGENCY" && worst && <EmergencyBanner worst={worst} onAckAll={ackAll} />}

      <main className="relative z-10 flex-1 min-h-0 p-5 grid grid-cols-[320px_minmax(0,1fr)_320px] gap-5">
        <LeftPanel live={live} selected={selected} onSelect={setSelected} canReport={!!live && !!s.hello?.muse}
          reportBusy={reportBusy} onReport={() => { setReportBusy(true); s.send({ cmd: "report" }) }} />

        <section className="flex flex-col gap-4 min-w-0 min-h-0">
          <Feed connected={s.connected} hello={s.hello} status={s.status} live={live} frame={s.frame}
            selected={selected} onSelect={setSelected}
            showOutline={showOutline} setShowOutline={setShowOutline} showUnv={showUnv} setShowUnv={setShowUnv}
            source={source} setSource={setSource} onStart={start}
            calibrating={calibrating} calib={s.calib}
            onRecalibrate={() => { s.setCalib(null); setCalibrating(true); s.send({ cmd: "grab_frame", source }) }}
            onSaveZone={(points) => { s.send({ cmd: "save_zone", points, jpeg: s.calib }); setCalibrating(false) }}
            onCancelCalib={() => setCalibrating(false)}
            onCancel={() => s.send({ cmd: "stop" })} onBack={() => s.send({ cmd: "stop" })}
            onBegin={() => { unlockSound(); s.send({ cmd: "begin" }) }} />
          {live && (
            <div className="-mt-2 flex justify-end">
              <button onClick={() => s.send({ cmd: "stop" })} className="text-[12px] text-neutral-500 hover:text-neutral-200">Stop monitoring</button>
            </div>
          )}
          {live && <Events events={live.events} />}
        </section>

        <Swimmers swimmers={live?.swimmers ?? null} selected={selected} onSelect={setSelected}
          onAck={(id) => s.send({ cmd: "ack", id })} thresholds={s.hello?.thresholds ?? {}} />
      </main>

      {s.error && (
        <div className="fixed bottom-4 right-4 z-[60] max-w-[420px] rounded-xl bg-red-700 text-white px-4 py-3 flex gap-3 items-start glow-red">
          <div className="text-[14px]">{s.error}</div>
          <button onClick={() => s.setError(null)} className="ml-auto"><X className="size-4" /></button>
        </div>
      )}

      <Dialog open={!!s.report} onOpenChange={(o) => !o && s.setReport(null)}>
        <DialogContent className="max-w-2xl bg-neutral-950 border-white/10">
          <DialogHeader><DialogTitle>Muse Spark session report</DialogTitle></DialogHeader>
          <div className="max-h-[60vh] overflow-y-auto whitespace-pre-wrap text-[14px] leading-relaxed text-neutral-200">{s.report}</div>
        </DialogContent>
      </Dialog>
    </div>
  )
}
