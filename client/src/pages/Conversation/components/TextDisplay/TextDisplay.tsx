import { FC, useEffect, useMemo, useRef } from "react";
import { ToolStatus, useServerText } from "../../hooks/useServerText";

type TextDisplayProps = {
  containerRef: React.RefObject<HTMLDivElement>;
};

export const TextDisplay:FC<TextDisplayProps> = ({
  containerRef,
}) => {
  const { text, toolEvents, toolStatuses, userTranscript } = useServerText();
  const currentIndex = text.length - 1;
  const currentUserIndex = userTranscript.length - 1;
  const prevScrollTop = useRef(0);
  const recentToolEvents = useMemo(() => toolEvents.slice(-8), [toolEvents]);
  const orderedToolStatuses = useMemo(
    () => Object.values(toolStatuses).sort((a, b) => a.updatedAt - b.updatedAt),
    [toolStatuses],
  );

  useEffect(() => {
    if (containerRef.current) {
      prevScrollTop.current = containerRef.current.scrollTop;
      containerRef.current.scroll({
        top: containerRef.current.scrollHeight,
        behavior: "smooth",
      });
    }
  }, [text, toolEvents, userTranscript]);

  const statusClass = (status: ToolStatus["status"]) => {
    if (status === "running") {
      return "bg-yellow-500/20 text-yellow-200 border-yellow-400/40";
    }
    if (status === "completed") {
      return "bg-green-500/20 text-green-200 border-green-400/40";
    }
    return "bg-red-500/20 text-red-200 border-red-400/40";
  };

  return (
    <div className="h-full w-full max-w-full max-h-full p-2">
      <div className="flex h-full flex-col gap-3">
        <div className="grid grid-cols-1 gap-3 md:grid-cols-[280px_minmax(0,1fr)_minmax(0,1fr)]">
          <div className="rounded-md border border-white/20 bg-black/20 p-2 text-xs">
            <div className="mb-2 font-semibold tracking-wide uppercase opacity-80">Tool activity</div>
            {recentToolEvents.length === 0 && (
              <div className="opacity-70">No tool activity yet.</div>
            )}
            {recentToolEvents.length > 0 && (
              <div className="flex flex-col gap-1">
                {recentToolEvents.map((event, i) => (
                  <div key={`${event.tool}-${event.phase}-${event.timestamp}-${i}`} className="rounded border border-white/10 bg-black/20 p-2">
                    <div className="flex items-center justify-between gap-2">
                      <div>{event.tool}</div>
                      <div className="opacity-70">{event.phase}</div>
                    </div>
                    <div className="opacity-60">{new Date(event.timestamp).toLocaleTimeString()}</div>
                  </div>
                ))}
              </div>
            )}
          </div>
          <div className="rounded-md border border-white/20 bg-black/10 p-2 min-h-[120px]">
            <div className="mb-2 text-xs font-semibold tracking-wide uppercase opacity-80">Assistant speech</div>
            {text.map((t, i) => (
              <span
                key={i}
                className={`${i === currentIndex ? "font-bold" : "font-normal"}`}
              >
                {t}
              </span>
            ))}
          </div>
          <div className="rounded-md border border-white/20 bg-black/10 p-2 min-h-[120px]">
            <div className="mb-2 text-xs font-semibold tracking-wide uppercase opacity-80">User speech</div>
            {userTranscript.length === 0 && (
              <div className="text-sm opacity-60">Waiting for user transcript...</div>
            )}
            {userTranscript.map((entry, i) => (
              <span
                key={`${entry.timestamp}-${i}`}
                className={`${i === currentUserIndex ? "font-bold" : "font-normal"} text-sm`}
              >
                {entry.text}
              </span>
            ))}
          </div>
        </div>
        <div className="rounded-md border border-white/20 bg-black/20 p-2 text-xs">
          <div className="mb-2 font-semibold tracking-wide uppercase opacity-80">Tool status bar</div>
          {orderedToolStatuses.length === 0 && (
            <div className="opacity-70">No tool status yet.</div>
          )}
          {orderedToolStatuses.length > 0 && (
            <div className="flex gap-2 overflow-x-auto pb-1">
              {orderedToolStatuses.map((status) => {
                const details = status.status === "running"
                  ? status.triggerText
                  : status.status === "completed" || status.status === "failed"
                    ? `rc=${status.returnCode ?? "?"}${status.durationMs ? ` • ${Math.round(status.durationMs)}ms` : ""}${status.stdout ? ` • ${status.stdout}` : ""}`
                    : status.error;

                return (
                  <div
                    key={`status-${status.tool}`}
                    className="min-w-[220px] rounded border border-white/10 bg-black/20 p-2"
                  >
                    <div className="flex items-center justify-between gap-2">
                      <div className="font-medium">{status.tool}</div>
                      <div className={`rounded border px-2 py-0.5 text-[10px] uppercase tracking-wide ${statusClass(status.status)}`}>
                        {status.status}
                      </div>
                    </div>
                    <div className="mt-1 opacity-70">{new Date(status.updatedAt).toLocaleTimeString()}</div>
                    {details && <div className="mt-1 break-words opacity-90">{details}</div>}
                  </div>
                );
              })}
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
