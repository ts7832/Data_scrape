import { HTMLTable, Section, SectionCard, Tag } from "@blueprintjs/core";
import type { ImpulseEvent } from "../api/types";
import { isoZ, shortImpulseId, shortTrackId } from "../format";
import type { Action, State } from "../state/reducer";
import { IMPULSE_INTENT } from "../theme";

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return <tr><td>{label}</td><td>{children}</td></tr>;
}

function Summary({ event, dispatch }: { event: ImpulseEvent; dispatch: (a: Action) => void }) {
  const e = event.ellipse;
  return (
    <HTMLTable compact className="kv">
      <colgroup><col style={{ width: 88 }} /><col /></colgroup>
      <tbody>
        <Field label="KIND">{event.kind === "drone_impact" ? "DRONE IMPACT" : "UNASSOCIATED"}</Field>
        <Field label="QUALITY">{event.quality.toUpperCase()}</Field>
        <Field label="POS">{event.position.lat.toFixed(4)} {event.position.lon.toFixed(4)}</Field>
        <Field label="95% ELLIPSE">
          {Math.round(e.semi_major_m)} × {Math.round(e.semi_minor_m)} M @ {Math.round(e.bearing_deg)}°
        </Field>
        <Field label="TIME">{isoZ(event.occurred_at)}</Field>
        <Field label="SENSORS">
          {event.node_ids.length}
          {(event.excluded_node_ids?.length ?? 0) > 0 && (
            <span className="muted"> ({event.excluded_node_ids!.length} excluded: {" "}
              {event.excluded_node_ids!.join(", ")})</span>
          )}
        </Field>
        <Field label="RMS">
          {event.rms_residual_ms != null ? `${event.rms_residual_ms.toFixed(1)} MS` : "—"}
        </Field>
        <Field label="TRACK">
          {event.associated_track_id ? (
            <a onClick={() => dispatch({ type: "select", trackId: event.associated_track_id! })}>
              {shortTrackId(event.associated_track_id)}
            </a>
          ) : "—"}
        </Field>
        <Field label="ALERT">
          <a href={`/v1/impulse-events/${event.event_id}/cap`} target="_blank" rel="noreferrer">
            CAP XML
          </a>
        </Field>
      </tbody>
    </HTMLTable>
  );
}

function Residuals({ event }: { event: ImpulseEvent }) {
  const rows = Object.entries(event.residuals_ms ?? {});
  const excluded = new Set(event.excluded_node_ids ?? []);
  if (rows.length === 0) return null;
  return (
    <Section compact collapsible title="Residuals" rightElement={<Tag minimal>{rows.length} NODES</Tag>}>
      <SectionCard padded={false}>
        <HTMLTable compact striped>
          <colgroup><col /><col style={{ width: 72 }} /><col style={{ width: 64 }} /></colgroup>
          <thead><tr><th>NODE</th><th className="num">RESIDUAL</th><th>STATUS</th></tr></thead>
          <tbody>
            {rows.map(([node, ms]) => (
              <tr key={node}>
                <td>{node}</td>
                <td className="num">{ms.toFixed(1)} MS</td>
                <td className="muted">{excluded.has(node) ? "EXCLUDED" : "USED"}</td>
              </tr>
            ))}
          </tbody>
        </HTMLTable>
      </SectionCard>
    </Section>
  );
}

export function ImpulseDetail({ state, dispatch }: { state: State; dispatch: (a: Action) => void }) {
  const selected = state.selectedImpulseId;
  const event = selected ? state.impulses[selected] : undefined;

  return (
    <aside className="panel right">
      <Section compact
        title={event ? `Impulse ${shortImpulseId(event.event_id)}` : "Impulse"}
        rightElement={event && <Tag minimal intent={IMPULSE_INTENT[event.kind]}>
          {event.kind === "drone_impact" ? "IMPACT" : "IMPULSE"}
        </Tag>}>
        <SectionCard padded={false}>
          {event ? <Summary event={event} dispatch={dispatch} /> : (
            <HTMLTable compact><tbody><tr className="empty"><td>NO SELECTION</td></tr></tbody></HTMLTable>
          )}
        </SectionCard>
      </Section>
      {event && <Residuals event={event} />}
    </aside>
  );
}
