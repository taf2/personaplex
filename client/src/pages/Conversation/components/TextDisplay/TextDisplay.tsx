import { FC, useEffect, useMemo, useRef } from "react";
import { ToolStatus, useServerText } from "../../hooks/useServerText";

type TextDisplayProps = {
  containerRef: React.RefObject<HTMLDivElement>;
};

export const TextDisplay:FC<TextDisplayProps> = ({
  containerRef,
}) => {
  const { text, toolEvents, toolStatuses } = useServerText();
  const currentIndex = text.length - 1;
  const prevScrollTop = useRef(0);
  const recentToolEvents = useMemo(() => toolEvents.slice(-8).reverse(), [toolEvents]);
  const orderedToolStatuses = useMemo(
    () => Object.values(toolStatuses).sort((a, b) => b.updatedAt - a.updatedAt),
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
  }, [text, toolEvents]);

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
    <div className="h-full w-full max-w-full max-h-full  p-2">
      <div className="grid grid-cols-1 gap-3 md:grid-cols-[280px_minmax(0,1fr)]">
        <div className="rounded-md border border-white/20 bg-black/20 p-2 text-xs">
          <div className="mb-2 font-semibold tracking-wide uppercase opacity-80">Tool status</div>
          {orderedToolStatuses.length === 0 && (
            <div className="opacity-70">No tool activity yet.</div>
          )}
          {orderedToolStatuses.length > 0 && (
            <div className="flex flex-col gap-2">
              {orderedToolStatuses.map((status) => {
                const details = status.status === "running"
                  ? status.triggerText
                  : status.status === "completed" || status.status === "failed"
                    ? `rc=${status.returnCode ?? "?"}${status.durationMs ? ` • ${Math.round(status.durationMs)}ms` : ""}${status.stdout ? ` • ${status.stdout}` : ""}`
                    : status.error;

                return (
                  <div key={`status-${status.tool}`} className="rounded border border-white/10 bg-black/20 p-2">
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
          {recentToolEvents.length > 0 && (
            <div className="mt-3 border-t border-white/10 pt-2">
              <div className="mb-2 font-semibold tracking-wide uppercase opacity-80">Recent activity</div>
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
            </div>
          )}
        </div>
        <div className="rounded-md border border-white/20 bg-black/10 p-2 min-h-[120px]">
        {text.map((t, i) => (
          <span
            key={i}
            className={`${i === currentIndex ? "font-bold" : "font-normal"}`}
          >
            {t}
          </span>
        ))}
        </div>
      </div>
    </div>
  );
};
