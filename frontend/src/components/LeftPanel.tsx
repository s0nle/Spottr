import { Sparkles } from "lucide-react"
import { fmtClock, type LiveState } from "@/lib/spottr"

interface Props {
  live: LiveState | null
  selected: number | null
  onSelect: (id: number) => void
  onReport: () => void
  canReport: boolean
  reportBusy: boolean
}

const Dash = () => <span className="text-neutral-600">–</span>

export function LeftPanel({ live, selected, onSelect, onReport, canReport, reportBusy }: Props) {
  const c = live?.counts
  return (
    <aside className="flex flex-col gap-4 min-h-0 overflow-hidden">
      <div className="rounded-2xl glass p-5">
        <div className="text-[13px] font-semibold text-neutral-400 uppercase tracking-wide">In pool</div>
        <div className="text-[60px] font-extrabold leading-none mt-2 text-muse">{c ? c.pool : <Dash />}</div>
        {c && (
          <div className="mt-3 flex flex-wrap gap-1.5 text-[12px]">
            {c.under > 0 && <span className="rounded-full bg-amber-400/15 text-amber-300 px-2 py-0.5 font-medium">{c.under} under</span>}
            {c.above > 0 && <span className="rounded-full bg-muse/20 text-[#7fb2ff] px-2 py-0.5 font-medium">{c.above} above</span>}
            {c.outview > 0 && <span className="rounded-full bg-white/10 text-neutral-300 px-2 py-0.5 font-medium">{c.outview} out of view</span>}
            {c.lost > 0 && <span className="rounded-full bg-red-600/20 text-red-300 px-2 py-0.5 font-medium">{c.lost} lost</span>}
          </div>
        )}
      </div>

      <div className="rounded-2xl glass p-5">
        <div className="text-[13px] font-semibold text-neutral-400 uppercase tracking-wide">On deck</div>
        <div className="text-[60px] font-extrabold leading-none mt-2 text-white">{live ? live.deck.length : <Dash />}</div>
        {live && live.deck.length > 0 && (
          <div className="mt-3 flex flex-wrap gap-1.5 text-[12px]">
            {live.deck.map((d) => (
              <button key={d.id} onClick={() => onSelect(d.id)}
                className={`rounded-md border px-2 py-0.5 font-semibold ${selected === d.id ? "border-white bg-white text-black" : "border-white/20 text-neutral-200 hover:border-white/50"}`}>
                ID {d.id}
              </button>
            ))}
          </div>
        )}
      </div>

      <div className="rounded-2xl glass p-4 text-[14px] space-y-2.5">
        <Row label="Entered pool" value={c?.entries} />
        <Row label="Left pool" value={c?.exits} />
        <Row label="Unverified" value={c?.unverified} dim />
        <div className="pt-2 border-t border-white/10 flex justify-between">
          <span className="text-neutral-400">Session</span>
          <b className="font-mono">{live ? fmtClock(live.session_s) : <Dash />}</b>
        </div>
      </div>

      <div className="mt-auto rounded-2xl border border-dashed border-white/15 bg-white/[0.03] backdrop-blur-xl p-4 text-[13px] text-neutral-400">
        <div className="font-semibold text-neutral-200 mb-1 flex items-center gap-1.5"><Sparkles className="size-[14px] text-muse-soft" />Session report</div>
        Muse Spark summarises every alarm when the session ends.
        <button onClick={onReport} disabled={!canReport || reportBusy}
          className="mt-3 w-full rounded-lg bg-white/10 border border-white/15 text-white font-medium py-2 disabled:opacity-40 hover:bg-white/15">
          {reportBusy ? "Writing report…" : "Write report now"}
        </button>
      </div>
    </aside>
  )
}

function Row({ label, value, dim }: { label: string; value?: number; dim?: boolean }) {
  return (
    <div className="flex justify-between">
      <span className="text-neutral-400">{label}</span>
      <b className={`font-mono ${dim ? "text-neutral-500" : ""}`}>{value ?? <Dash />}</b>
    </div>
  )
}
