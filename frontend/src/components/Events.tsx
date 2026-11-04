import { EVENT_STYLE } from "./common"
import type { SpottrEvent } from "@/lib/spottr"

export function Events({ events }: { events: SpottrEvent[] | null }) {
  return (
    <div className="flex-1 rounded-2xl glass p-4 min-h-0 flex flex-col overflow-hidden">
      <div className="flex items-center mb-1">
        <div className="font-semibold">Events</div>
        <div className="ml-auto text-[12px] text-neutral-500">newest first</div>
      </div>
      {!events || events.length === 0 ? (
        <div className="text-[14px] text-neutral-500 py-6 text-center">Events appear here once monitoring starts.</div>
      ) : (
        <ul className="text-[13.5px] divide-y divide-white/10 overflow-y-auto min-h-0">
          {events.map((e, i) => {
            const [cls, Icon, label] = EVENT_STYLE[e.kind] ?? ["bg-white/10 text-neutral-200", EVENT_STYLE.NEW[1]]
            return (
              <li key={`${e.t}-${e.kind}-${i}`} className="py-[7px] flex gap-3 items-center">
                <span className="font-mono text-neutral-500 w-[66px] shrink-0">{e.wall}</span>
                <span className={`inline-flex items-center gap-1 rounded text-[10.5px] font-bold px-1.5 py-0.5 w-[108px] shrink-0 whitespace-nowrap ${cls}`}>
                  <Icon className="size-[11px]" />{label ?? e.kind}
                </span>
                <span className="truncate">{e.id !== null ? `ID ${e.id} — ` : ""}{e.text}</span>
              </li>
            )
          })}
        </ul>
      )}
    </div>
  )
}
