import { Check, EyeOff, ScanFace, Search, Sparkles, Timer, VolumeX } from "lucide-react"
import { LEVEL_BADGE, Mascot } from "./common"
import type { Swimmer } from "@/lib/spottr"

interface Props {
  swimmers: Swimmer[] | null
  selected: number | null
  onSelect: (id: number) => void
  onAck: (id: number) => void
  thresholds: Record<string, number>
}

const SEL = "border-2 !border-white glow-white"

export function Swimmers({ swimmers, selected, onSelect, onAck, thresholds }: Props) {
  return (
    <aside className="flex flex-col gap-3 min-h-0 overflow-hidden">
      <div className="flex items-center">
        <div className="font-semibold text-[15px]">Swimmers</div>
        <div className="ml-auto text-[12px] text-neutral-500">click a card to find them</div>
      </div>

      {!swimmers ? (
        <div className="rounded-2xl border border-dashed border-white/15 bg-white/[0.03] p-6 flex flex-col items-center text-center">
          <Mascot className="h-[110px] w-auto opacity-90" />
          <div className="text-[15px] font-semibold text-neutral-200 mt-2">No swimmers yet</div>
          <div className="text-[13px] text-neutral-500 mt-1">The lifeguard is ready. Swimmers appear here once monitoring starts.</div>
        </div>
      ) : swimmers.length === 0 ? (
        <div className="rounded-2xl border border-dashed border-white/15 bg-white/[0.03] p-6 flex flex-col items-center text-center">
          <Mascot className="h-[90px] w-auto opacity-80" />
          <div className="text-[14px] text-neutral-400 mt-2">Nobody in the pool right now.</div>
        </div>
      ) : (
        <div className="flex flex-col gap-3 min-h-0 overflow-y-auto pr-1">
          {swimmers.map((w) => {
            const sel = selected === w.id
            const click = () => onSelect(w.id)
            if (w.state === "LOST") return <LostCard key={w.id} w={w} sel={sel} onClick={click} onAck={onAck} />
            if (w.state === "OUTVIEW") return <OutCard key={w.id} w={w} sel={sel} onClick={click} t={thresholds} />
            const under = w.alarm !== "OK" || ((w.state === "SUBMERGED" || w.state === "UNSEEN") && w.under_s >= 3)
            return under
              ? <UnderCard key={w.id} w={w} sel={sel} onClick={click} onAck={onAck} t={thresholds} />
              : <OkCard key={w.id} w={w} sel={sel} onClick={click} />
          })}
        </div>
      )}

      {(!swimmers || swimmers.length <= 2) && (
        <div className="mt-auto rounded-2xl glass px-4 py-3 text-[11.5px] text-neutral-400 grid grid-cols-2 gap-x-3 gap-y-1.5">
          <div className="flex gap-1.5 items-center"><span className="size-2.5 rounded-full bg-amber-400" />WATCH · {thresholds.watch_under ?? 10} s</div>
          <div className="flex gap-1.5 items-center"><span className="size-2.5 rounded-full bg-red-600" />ALARM · {thresholds.alarm_under ?? 20} s</div>
          <div className="flex gap-1.5 items-center"><span className="size-2.5 rounded-full bg-red-900 ring-1 ring-red-400" />EMERGENCY · {thresholds.emergency_under ?? 30} s</div>
          <div className="flex gap-1.5 items-center"><span className="size-2.5 rounded-sm border-2 border-white" />selected</div>
        </div>
      )}
    </aside>
  )
}

function HeadLine({ w }: { w: Swimmer }) {
  if (!w.head_known) return null
  return w.head_up
    ? <span className="flex items-center gap-1 text-[#7fb2ff]"><ScanFace className="size-[15px]" />Head up</span>
    : <span className="flex items-center gap-1.5 text-amber-300"><ScanFace className="size-[15px]" />Head not seen</span>
}

function StillLine({ w, dim }: { w: Swimmer; dim?: boolean }) {
  if (w.still_s < 3) return null
  return (
    <span className={`flex items-center gap-1.5 ${dim ? "text-neutral-400" : "text-neutral-300"}`}>
      <Timer className="size-[15px]" />{dim ? "still" : "Not moving"} <b className="font-mono">{Math.round(w.still_s)} s</b>
    </span>
  )
}

function MuseBox({ w }: { w: Swimmer }) {
  const m = w.muse
  if (!m) return null
  return (
    <div className="mt-2.5 rounded-lg bg-black/40 border border-white/10 px-3 py-2 text-[12.5px] text-neutral-300">
      <div className="text-[11px] text-neutral-500 mb-0.5 flex items-center gap-1">
        <Sparkles className="size-[12px] text-muse-soft" />Muse Spark · asked at {Math.round(m.asked_under ?? m.asked_t)} s
        {m.answer_s !== undefined && <> · {Math.round(m.answer_s)} s to answer</>}
      </div>
      {m.verdict
        ? <span>{m.verdict} — {m.reason}{m.confidence !== undefined && <> ({Number(m.confidence).toFixed(2)})</>}{m.pending && <span className="text-neutral-500"> · re-checking…</span>}</span>
        : m.pending ? <span className="text-neutral-400">asking for a second opinion…</span>
        : <span className="text-neutral-500">no answer</span>}
    </div>
  )
}

function UnderCard({ w, sel, onClick, onAck, t }: { w: Swimmer; sel: boolean; onClick: () => void; onAck: (id: number) => void; t: Record<string, number> }) {
  // which timer drives this alarm (a Muse-raised alarm has no number: use the larger timer)
  const stillBased = w.reason.startsWith("still") || (w.reason.startsWith("Muse") && w.still_s > w.under_s)
  // the alarm's own number (a level is held for 3 s after its cause flickers,
  // so the live timer can briefly read 0 while the card is still at WATCH)
  const fromReason = Number(/(\d+)s/.exec(w.reason)?.[1] ?? NaN)
  const live = stillBased ? w.still_s : w.under_s
  const secs = w.alarm !== "OK" && !Number.isNaN(fromReason) ? Math.max(live, fromReason) : live
  const max = stillBased ? (t.still_alarm ?? 20) : (t.alarm_under ?? 20)
  const red = w.alarm === "ALARM" || w.alarm === "EMERGENCY"
  const em = w.alarm === "EMERGENCY"
  const frac = red ? 1 : Math.min(1, secs / max)
  const C = 2 * Math.PI * 52
  const box = em ? "border-2 border-red-500 glow-red" : red ? "border-2 border-red-500 glow-red" : "border-2 border-amber-400/70"
  const tint = em ? "rgba(220,38,38,.22)" : red ? "rgba(220,38,38,.14)" : "rgba(245,158,11,.06)"
  return (
    <div onClick={onClick} className={`cursor-pointer rounded-2xl glass p-3.5 ${sel ? SEL : box}`} style={{ background: tint }}>
      <div className="flex gap-3.5 items-center">
        <div className="relative size-[104px] shrink-0">
          <svg viewBox="0 0 120 120" className="size-full -rotate-90">
            <circle cx="60" cy="60" r="52" fill="none" stroke="#262626" strokeWidth="10" />
            <circle cx="60" cy="60" r="52" fill="none" stroke={em ? "#991b1b" : red ? "#dc2626" : "#f59e0b"} strokeWidth="10"
              strokeLinecap="round" strokeDasharray={C} strokeDashoffset={C * (1 - frac)} />
          </svg>
          <div className={`absolute inset-[16px] rounded-full bg-gradient-to-b flex flex-col items-center justify-center ${em ? "from-[#3b0606] to-red-700" : red ? "from-red-900 to-red-500" : "from-[#0b2a5c] to-[#0866FF]"}`}>
            <span className="absolute top-[12px] left-[18px] size-2 rounded-full border border-white/80" />
            <span className="absolute top-[21px] right-[15px] size-1.5 rounded-full border border-white/80" />
            <span className="absolute top-[6px] right-[26px] size-1 rounded-full bg-white/70" />
            <div className="font-mono text-[26px] font-bold text-white leading-none">{Math.round(secs)}s</div>
            <div className="text-[10px] font-semibold text-white/80 mt-1">{stillBased ? "not moving" : "under water"}</div>
          </div>
        </div>
        <div className="min-w-0 flex-1">
          <div className="flex items-center gap-2">
            <div className="text-[19px] font-extrabold">ID {w.id}</div>
            <span className={`rounded-full text-[11px] font-bold px-2 py-0.5 ${LEVEL_BADGE[w.alarm]}`}>{w.alarm === "OK" ? "UNDER" : w.alarm}</span>
          </div>
          <div className="mt-1.5 space-y-1 text-[13px]">
            <HeadLine w={w} />
            {!stillBased && <StillLine w={w} />}
            {em ? <div className="font-bold text-red-300">Go now — {Math.round(w.under_s || secs)} s</div>
              : w.next && <div className="text-neutral-500">{w.next.level === "EMERGENCY" ? "Emergency" : w.next.level === "ALARM" ? "Alarm" : "Watch"} in <b className="font-mono text-neutral-200">{Math.ceil(w.next.in_s)} s</b></div>}
          </div>
          {w.alarm !== "OK" && (w.acked
            ? <span className="mt-2 inline-flex items-center gap-1.5 rounded-md bg-white/10 border border-white/15 text-neutral-300 text-[12px] font-semibold px-2 py-0.5"><VolumeX className="size-[13px]" />Acknowledged · sound off</span>
            : <button onClick={(e) => { e.stopPropagation(); onAck(w.id) }}
                className={`mt-2 rounded-lg text-[12.5px] font-bold px-3 py-1 ${red ? "bg-white text-red-700" : "bg-white/10 border border-white/15 text-white"}`}>Acknowledge</button>)}
        </div>
      </div>
      <MuseBox w={w} />
    </div>
  )
}

function LostCard({ w, sel, onClick, onAck }: { w: Swimmer; sel: boolean; onClick: () => void; onAck: (id: number) => void }) {
  return (
    <div onClick={onClick} className={`cursor-pointer rounded-2xl glass p-3.5 flex items-center gap-4 ${sel ? SEL : "border-2 !border-red-500/80"}`} style={{ background: "rgba(220,38,38,.10)" }}>
      <div className="relative size-[72px] shrink-0 rounded-full border-[3px] border-dashed border-red-400 flex flex-col items-center justify-center">
        <Search className="size-[20px] text-red-300" />
        <div className="font-mono text-[13px] font-bold text-red-200 mt-0.5">{Math.round(w.seen_ago)}s</div>
      </div>
      <div className="min-w-0">
        <div className="flex items-center gap-2"><div className="text-[18px] font-extrabold">ID {w.id}</div>
          <span className="rounded-full bg-red-600 text-white text-[11px] font-bold px-2 py-0.5">LOST</span></div>
        <div className="text-[13.5px] text-neutral-300 mt-1">{w.lost_at_edge ? `Last seen at the ${w.side} edge` : "Last seen in the pool"}</div>
        <div className="text-[12px] text-neutral-500">{Math.round(w.seen_ago)} s ago{w.lost_at_edge ? " · may have swum out of view" : " · not clearly seen since"}</div>
        {w.acked
          ? <span className="mt-2 inline-flex items-center gap-1.5 rounded-md bg-white/10 border border-white/15 text-neutral-300 text-[12px] font-semibold px-2 py-0.5"><VolumeX className="size-[13px]" />Acknowledged · sound off</span>
          : <button onClick={(e) => { e.stopPropagation(); onAck(w.id) }} className="mt-2 rounded-lg bg-white text-red-700 text-[12px] font-bold px-3 py-1">Checked · dismiss</button>}
      </div>
    </div>
  )
}

function OutCard({ w, sel, onClick, t }: { w: Swimmer; sel: boolean; onClick: () => void; t: Record<string, number> }) {
  return (
    <div onClick={onClick} className={`cursor-pointer rounded-2xl glass p-3.5 flex items-center gap-4 ${sel ? SEL : "!border-dashed !border-white/35"}`}>
      <div className="relative size-[64px] shrink-0 rounded-full border-[3px] border-dashed border-white/50 flex flex-col items-center justify-center">
        <EyeOff className="size-[20px] text-neutral-200" />
        <div className="font-mono text-[12px] font-bold text-neutral-200 mt-0.5">{Math.round(w.under_s)}s</div>
      </div>
      <div className="min-w-0">
        <div className="flex items-center gap-2"><div className="text-[17px] font-extrabold">ID {w.id}</div>
          <span className={`rounded-full text-[11px] font-bold px-2 py-0.5 ${w.alarm === "WATCH" ? LEVEL_BADGE.WATCH : "bg-white text-black"}`}>{w.alarm === "WATCH" ? "WATCH" : "OUT OF VIEW"}</span></div>
        <div className="text-[13.5px] text-neutral-300 mt-1">Swam past the {w.side} edge</div>
        <div className="text-[12px] text-neutral-500">WATCH at {t.outview_watch ?? 20} s · LOST at 60 s</div>
      </div>
    </div>
  )
}

function OkCard({ w, sel, onClick }: { w: Swimmer; sel: boolean; onClick: () => void }) {
  const museOk = w.muse?.verdict === "ok" && !w.muse.pending
  return (
    <div onClick={onClick} className={`cursor-pointer rounded-2xl glass p-3.5 ${sel ? SEL : ""}`}>
      <div className="flex gap-4 items-center">
        <div className="relative size-[64px] shrink-0">
          <svg viewBox="0 0 120 120" className="size-full"><circle cx="60" cy="60" r="52" fill="none" stroke="#0866FF" strokeWidth="10" /></svg>
          <div className="absolute inset-[11px] rounded-full bg-muse/15 flex items-center justify-center"><Check className="size-[22px] text-muse-soft" /></div>
        </div>
        <div className="min-w-0">
          <div className="flex items-center gap-2"><div className="text-[17px] font-extrabold">ID {w.id}</div>
            <span className={`rounded-full text-[11px] font-bold px-2 py-0.5 ${LEVEL_BADGE.OK}`}>OK</span>
            {museOk && <span className="rounded-full border border-white/20 text-neutral-400 text-[10.5px] font-semibold px-1.5 py-0.5 line-through">WATCH</span>}
          </div>
          <div className="mt-1 flex items-center gap-3 text-[12.5px]">
            <HeadLine w={w} />
            {w.still_s >= 3 ? <StillLine w={w} dim /> : <span className="text-neutral-400">swimming</span>}
          </div>
        </div>
        {sel && <div className="ml-auto self-start rounded-md bg-white text-black text-[11px] font-bold px-1.5 py-0.5">SELECTED</div>}
      </div>
      {museOk && (
        <div className="mt-2.5 rounded-lg bg-black/40 border border-white/10 px-3 py-1.5 text-[12.5px] text-neutral-300">
          <span className="text-muse-soft font-semibold inline-flex items-center gap-1"><Sparkles className="size-[12px]" />Muse:</span> {w.muse!.reason} — watch cancelled
        </div>
      )}
    </div>
  )
}
