import { useEffect, useState } from "react";
import { Intent, NonIdealState, Tag } from "@blueprintjs/core";
import { fetchTrackDetail } from "../api/client";
import type { Track, TrackDetail as Detail } from "../api/types";
import { clockUtc, percent, shortTrackId } from "../format";
import type { State } from "../state/reducer";

const INTENT: Record<string, Intent> = {
  confirmed: Intent.DANGER, tentative: Intent.WARNING, downgraded: Intent.NONE, closed: Intent.NONE,
};

const time = (iso: string) => clockUtc(Date.parse(iso));

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <>
      <dt className="caps muted">{label}</dt>
      <dd className="mono">{children}</dd>
    </>
  );
}

function Summary({ track }: { track: Track }) {
  return (
    <dl className="fields">
      <Field label="Confidence">{percent(track.confidence)}</Field>
      <Field label="Accuracy">±{Math.round(track.uncertainty_m)} m</Field>
      <Field label="Class">{track.label.replace(/_/g, " ")}</Field>
      <Field label="Position">{track.position.lat.toFixed(4)}, {track.position.lon.toFixed(4)}</Field>
      <Field label="Speed">{track.velocity ? `${track.velocity.speed_mps.toFixed(1)} m/s` : "—"}</Field>
      <Field label="Heading">{track.velocity ? `${Math.round(track.velocity.heading_deg)}°` : "—"}</Field>
      <Field label="First seen">{time(track.first_seen)}</Field>
      <Field label="Last seen">{time(track.last_seen)}</Field>
    </dl>
  );
}

function Evidence({ detail }: { detail: Detail }) {
  const nodes = new Set(detail.observations.map((o) => o.source.id));
  return (
    <>
      <h2 className="caps">
        Evidence <span className="mono muted">{detail.observations.length} obs · {nodes.size} nodes</span>
      </h2>
      <ul>
        {detail.observations.slice(-12).reverse().map((o) => (
          <li key={o.observation_id} className="row">
            <span className="mono muted">{time(o.observed_at)}</span>
            <span className="mono grow">{o.source.id}</span>
            <span className="mono muted">{o.event.phase.toUpperCase()}</span>
            <span className="mono">{percent(o.detection.confidence)}</span>
            {o.acoustic && <span className="mono muted">{o.acoustic.snr_db.toFixed(0)} dB</span>}
          </li>
        ))}
      </ul>
      {detail.silent_neighbours.length > 0 && (
        <>
          <h2 className="caps">Silent neighbours</h2>
          <p className="small muted">
            Online nodes within hearing range that heard nothing; each lowers confidence.
          </p>
          <div className="tags">
            {detail.silent_neighbours.map((n) => (
              <Tag key={n.node_id} minimal className="mono">{n.node_id}</Tag>
            ))}
          </div>
        </>
      )}
    </>
  );
}

export function TrackDetail({ state }: { state: State }) {
  const [detail, setDetail] = useState<Detail | null>(null);
  const selected = state.selectedTrackId;
  const track = selected ? state.tracks[selected] : undefined;

  useEffect(() => {
    if (!selected) { setDetail(null); return; }
    let cancelled = false;
    fetchTrackDetail(selected).then((d) => { if (!cancelled) setDetail(d); }).catch(() => setDetail(null));
    return () => { cancelled = true; };
  }, [selected, track?.last_seen]);

  if (!track) {
    return (
      <aside className="panel right">
        <NonIdealState icon="locate" title="No track selected"
          description="Select a track on the map or in the track list." />
      </aside>
    );
  }

  return (
    <aside className="panel right">
      <header className="detail-head">
        <span className="callsign mono">{shortTrackId(track.track_id)}</span>
        <Tag intent={INTENT[track.status]} large>{track.status.toUpperCase()}</Tag>
      </header>
      <Summary track={track} />
      {detail && <Evidence detail={detail} />}
    </aside>
  );
}
