import { useEffect, useState } from "react";
import { fetchTrackDetail } from "../api/client";
import type { TrackDetail } from "../api/types";
import type { Action, State } from "../state/reducer";

const ORDER: Record<string, number> = { confirmed: 0, tentative: 1, downgraded: 2 };
const time = (iso?: string | null) => (iso ? new Date(iso).toLocaleTimeString() : "never");

export function SidePanel({ state, dispatch }: { state: State; dispatch: (a: Action) => void }) {
  const [detail, setDetail] = useState<TrackDetail | null>(null);
  const selected = state.selectedTrackId;
  const selectedTrack = selected ? state.tracks[selected] : undefined;

  useEffect(() => {
    if (!selected) { setDetail(null); return; }
    let cancelled = false;
    fetchTrackDetail(selected).then((d) => { if (!cancelled) setDetail(d); }).catch(() => setDetail(null));
    return () => { cancelled = true; };
  }, [selected, selectedTrack?.last_seen]);

  const tracks = Object.values(state.tracks).sort(
    (a, b) => (ORDER[a.status] ?? 9) - (ORDER[b.status] ?? 9) || b.confidence - a.confidence);

  return (
    <aside className="panel">
      <header>
        <h1>Kuulo</h1>
        <span className={`conn conn-${state.connection}`}>{state.connection}</span>
      </header>

      <section>
        <h2>Tracks ({tracks.length})</h2>
        {tracks.length === 0 && <p className="muted">No active tracks</p>}
        <ul>
          {tracks.map((t) => (
            <li key={t.track_id} className={t.track_id === selected ? "selected" : ""}
                onClick={() => dispatch({ type: "select", trackId: t.track_id })}>
              <span className={`badge badge-${t.status}`}>{t.status}</span>
              {t.label} · {(t.confidence * 100).toFixed(0)}% · ±{Math.round(t.uncertainty_m)} m
            </li>
          ))}
        </ul>
      </section>

      {detail && (
        <section>
          <h2>Evidence</h2>
          <p>{detail.observations.length} observations from{" "}
            {new Set(detail.observations.map((o) => o.source.id)).size} node(s)</p>
          <ul className="evidence">
            {detail.observations.slice(-12).map((o) => (
              <li key={o.observation_id}>
                {time(o.observed_at)} · {o.source.id} · {o.event.phase} · {(o.detection.confidence * 100).toFixed(0)}%
                {o.acoustic ? ` · SNR ${o.acoustic.snr_db.toFixed(1)} dB` : ""}
              </li>
            ))}
          </ul>
          {detail.silent_neighbours.length > 0 && (
            <p className="muted">Silent online neighbours lowered confidence:{" "}
              {detail.silent_neighbours.map((n) => n.node_id).join(", ")}</p>
          )}
        </section>
      )}

      <section>
        <h2>Nodes ({Object.keys(state.nodes).length})</h2>
        <ul>
          {Object.values(state.nodes).map((n) => (
            <li key={n.node_id}>
              <span className={`dot dot-${n.status}`} /> {n.node_id} · {n.status} · last {time(n.last_heartbeat_at)}
              {n.uncorroborated_rate_24h != null &&
                ` · ${(n.uncorroborated_rate_24h * 100).toFixed(0)}% uncorroborated`}
            </li>
          ))}
        </ul>
      </section>
    </aside>
  );
}
