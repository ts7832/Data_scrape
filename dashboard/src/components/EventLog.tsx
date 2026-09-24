import { clockUtc } from "../format";
import type { LogEntry } from "../state/reducer";

export function EventLog({ events }: { events: LogEntry[] }) {
  return (
    <section className="eventlog">
      <h2 className="caps">Event log <span className="mono muted">{events.length}</span></h2>
      {events.length === 0 ? (
        <p className="muted small">Waiting for events…</p>
      ) : (
        <ol>
          {events.map((e) => (
            <li key={e.seq} className="mono">
              <span className="muted">{clockUtc(e.at)}</span>
              <span className="source">{e.source}</span>
              <span className={`tone-${e.tone}`}>{e.message}</span>
            </li>
          ))}
        </ol>
      )}
    </section>
  );
}
