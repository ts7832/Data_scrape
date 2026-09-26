import { HTMLTable, Intent, Section, SectionCard, Tag, Tooltip } from "@blueprintjs/core";
import type { ImpulseEvent, NodeView, Track } from "../api/types";
import { isoZ, percent, shortImpulseId, shortTrackId } from "../format";
import type { Action, State } from "../state/reducer";
import { IMPULSE_INTENT, NODE_INTENT, TRACK_INTENT } from "../theme";

const ORDER: Record<string, number> = { confirmed: 0, tentative: 1, downgraded: 2 };

// Both tables share one column grid so their edges line up down the panel:
// name (flexible) | value | status.
function Cols() {
  return <colgroup><col /><col style={{ width: 84 }} /><col style={{ width: 112 }} /></colgroup>;
}

function EmptyRow({ text }: { text: string }) {
  return <tr className="empty"><td colSpan={3}>{text}</td></tr>;
}

function TrackRow({ track, selected, onSelect }: {
  track: Track; selected: boolean; onSelect: () => void;
}) {
  return (
    <tr className={selected ? "selected" : undefined} onClick={onSelect}>
      <td>{shortTrackId(track.track_id)}</td>
      <td className="num">{percent(track.confidence)}</td>
      <td><Tag minimal intent={TRACK_INTENT[track.status]}>{track.status.toUpperCase()}</Tag></td>
    </tr>
  );
}

function ImpulseRow({ event, selected, onSelect }: {
  event: ImpulseEvent; selected: boolean; onSelect: () => void;
}) {
  return (
    <tr className={selected ? "selected" : undefined} onClick={onSelect}>
      <td>{shortImpulseId(event.event_id)}</td>
      <td className="num">±{Math.round(event.ellipse.semi_major_m)} M</td>
      <td>
        <Tag minimal intent={IMPULSE_INTENT[event.kind]}>
          {event.kind === "drone_impact" ? "IMPACT" : "IMPULSE"}
        </Tag>
      </td>
    </tr>
  );
}

function NodeRow({ node }: { node: NodeView }) {
  const noisy = node.uncorroborated_rate_24h != null && node.uncorroborated_rate_24h > 0.5;
  return (
    <tr>
      <td>
        {node.node_id}
        {noisy && (
          <Tooltip compact placement="top" content={`UNCORROBORATED 24H ${percent(node.uncorroborated_rate_24h!)}`}>
            <Tag minimal intent={Intent.WARNING} className="cell-tag">NOISY</Tag>
          </Tooltip>
        )}
      </td>
      <td className="num">{node.last_heartbeat_at ? isoZ(node.last_heartbeat_at) : "--:--:--Z"}</td>
      <td><Tag minimal intent={NODE_INTENT[node.status]}>{node.status.toUpperCase()}</Tag></td>
    </tr>
  );
}

export function NodePanel({ state, dispatch }: { state: State; dispatch: (a: Action) => void }) {
  const nodes = Object.values(state.nodes).sort((a, b) => a.node_id.localeCompare(b.node_id));
  const online = nodes.filter((n) => n.status === "online").length;
  const tracks = Object.values(state.tracks).sort(
    (a, b) => (ORDER[a.status] ?? 9) - (ORDER[b.status] ?? 9) || b.confidence - a.confidence);
  const impulses = Object.values(state.impulses).sort((a, b) => b.updated_at.localeCompare(a.updated_at));

  return (
    <aside className="panel left">
      <Section compact collapsible title="Tracks" rightElement={<Tag minimal>{tracks.length}</Tag>}>
        <SectionCard padded={false}>
          <HTMLTable compact interactive>
            <Cols />
            <thead><tr><th>ID</th><th className="num">CONF</th><th>STATUS</th></tr></thead>
            <tbody>
              {tracks.length === 0 ? <EmptyRow text="NO ACTIVE TRACKS" /> : tracks.map((t) => (
                <TrackRow key={t.track_id} track={t} selected={t.track_id === state.selectedTrackId}
                  onSelect={() => dispatch({ type: "select", trackId: t.track_id })} />
              ))}
            </tbody>
          </HTMLTable>
        </SectionCard>
      </Section>
      <Section compact collapsible title="Impulses" rightElement={<Tag minimal>{impulses.length}</Tag>}>
        <SectionCard padded={false}>
          <HTMLTable compact interactive>
            <Cols />
            <thead><tr><th>ID</th><th className="num">±95%</th><th>KIND</th></tr></thead>
            <tbody>
              {impulses.length === 0 ? <EmptyRow text="NO IMPULSES" /> : impulses.map((e) => (
                <ImpulseRow key={e.event_id} event={e} selected={e.event_id === state.selectedImpulseId}
                  onSelect={() => dispatch({ type: "selectImpulse", id: e.event_id })} />
              ))}
            </tbody>
          </HTMLTable>
        </SectionCard>
      </Section>
      <Section compact collapsible title="Sensors" rightElement={<Tag minimal>{online}/{nodes.length}</Tag>}>
        <SectionCard padded={false}>
          <HTMLTable compact striped>
            <Cols />
            <thead><tr><th>NODE</th><th className="num">LAST HB</th><th>STATUS</th></tr></thead>
            <tbody>
              {nodes.length === 0 ? <EmptyRow text="NO NODES" /> :
                nodes.map((n) => <NodeRow key={n.node_id} node={n} />)}
            </tbody>
          </HTMLTable>
        </SectionCard>
      </Section>
    </aside>
  );
}
