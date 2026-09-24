import { useEffect, useState } from "react";
import { HTMLTable, ProgressBar, Section, SectionCard, Tag, Tooltip } from "@blueprintjs/core";
import { fetchTrackDetail } from "../api/client";
import type { Track, TrackDetail as Detail } from "../api/types";
import { isoZ, percent, shortTrackId } from "../format";
import type { State } from "../state/reducer";
import { TRACK_INTENT } from "../theme";

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return <tr><td>{label}</td><td>{children}</td></tr>;
}

function Summary({ track }: { track: Track }) {
  const v = track.velocity;
  return (
    <HTMLTable compact className="kv">
      <colgroup><col style={{ width: 88 }} /><col /></colgroup>
      <tbody>
        <Field label="CONF">
          <span className="kv-bar">
            <ProgressBar value={track.confidence} intent={TRACK_INTENT[track.status]} stripes={false} animate={false} />
            {percent(track.confidence)}
          </span>
        </Field>
        <Field label="CEP">±{Math.round(track.uncertainty_m)} M</Field>
        <Field label="CLASS">{track.label.replace(/_/g, " ").toUpperCase()}</Field>
        <Field label="POS">{track.position.lat.toFixed(4)} {track.position.lon.toFixed(4)}</Field>
        <Field label="SPD">{v ? `${v.speed_mps.toFixed(1)} M/S` : "—"}</Field>
        <Field label="HDG">{v ? `${String(Math.round(v.heading_deg)).padStart(3, "0")}°` : "—"}</Field>
        <Field label="T-INIT">{isoZ(track.first_seen)}</Field>
        <Field label="T-LAST">{isoZ(track.last_seen)}</Field>
      </tbody>
    </HTMLTable>
  );
}

function Evidence({ detail }: { detail: Detail }) {
  const nodes = new Set(detail.observations.map((o) => o.source.id));
  return (
    <Section compact collapsible title="Evidence"
      rightElement={<Tag minimal>{detail.observations.length} OBS · {nodes.size} NODES</Tag>}>
      <SectionCard padded={false}>
        <HTMLTable compact striped>
          <colgroup>
            <col style={{ width: 80 }} /><col /><col style={{ width: 64 }} />
            <col style={{ width: 48 }} /><col style={{ width: 52 }} />
          </colgroup>
          <thead>
            <tr><th>TIME</th><th>NODE</th><th>PHASE</th><th className="num">CONF</th><th className="num">SNR</th></tr>
          </thead>
          <tbody>
            {detail.observations.slice(-12).reverse().map((o) => (
              <tr key={o.observation_id}>
                <td className="muted">{isoZ(o.observed_at)}</td>
                <td>{o.source.id}</td>
                <td className="muted">{o.event.phase.toUpperCase()}</td>
                <td className="num">{percent(o.detection.confidence)}</td>
                <td className="num muted">{o.acoustic ? `${o.acoustic.snr_db.toFixed(0)} DB` : "—"}</td>
              </tr>
            ))}
          </tbody>
        </HTMLTable>
      </SectionCard>
      {detail.silent_neighbours.length > 0 && (
        <SectionCard padded={false} className="pad">
          <Tooltip compact placement="left" content="IN RANGE · ONLINE · NO DETECTION · LOWERS CONF">
            <span className="caps muted">Silent neighbours</span>
          </Tooltip>
          <div className="tags" style={{ marginTop: 4 }}>
            {detail.silent_neighbours.map((n) => <Tag key={n.node_id} minimal>{n.node_id}</Tag>)}
          </div>
        </SectionCard>
      )}
    </Section>
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

  return (
    <aside className="panel right">
      <Section compact
        title={track ? `Track ${shortTrackId(track.track_id)}` : "Track"}
        rightElement={track && <Tag minimal intent={TRACK_INTENT[track.status]}>{track.status.toUpperCase()}</Tag>}>
        <SectionCard padded={false}>
          {track ? <Summary track={track} /> : (
            <HTMLTable compact><tbody><tr className="empty"><td>NO SELECTION</td></tr></tbody></HTMLTable>
          )}
        </SectionCard>
      </Section>
      {track && detail && <Evidence detail={detail} />}
    </aside>
  );
}
