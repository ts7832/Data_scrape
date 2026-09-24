import { Intent, Tag } from "@blueprintjs/core";
import type { NodeView, Track } from "../api/types";
import { clockUtc, percent, shortTrackId } from "../format";
import type { Action, State } from "../state/reducer";

const ORDER: Record<string, number> = { confirmed: 0, tentative: 1, downgraded: 2 };
const TRACK_INTENT: Record<string, Intent> = {
  confirmed: Intent.DANGER, tentative: Intent.WARNING, downgraded: Intent.NONE,
};

function NodeRow({ node }: { node: NodeView }) {
  const last = node.last_heartbeat_at ? clockUtc(Date.parse(node.last_heartbeat_at)) : "--:--:--";
  return (
    <li className="row">
      <span className={`dot dot-${node.status}`} />
      <span className="mono grow">{node.node_id}</span>
      <span className="mono muted">{last}</span>
      {node.uncorroborated_rate_24h != null && node.uncorroborated_rate_24h > 0.5 && (
        <Tag minimal intent={Intent.WARNING} className="gap-left" title="Share of this node's detections no other node confirmed (24 h)">
          NOISY
        </Tag>
      )}
    </li>
  );
}

function TrackRow({ track, selected, onSelect }: {
  track: Track; selected: boolean; onSelect: () => void;
}) {
  return (
    <li className={`row clickable ${selected ? "selected" : ""}`} onClick={onSelect}>
      <span className="mono grow">{shortTrackId(track.track_id)}</span>
      <span className="mono muted">{percent(track.confidence)}</span>
      <Tag minimal intent={TRACK_INTENT[track.status]} className="gap-left">
        {track.status.toUpperCase()}
      </Tag>
    </li>
  );
}

export function NodePanel({ state, dispatch }: { state: State; dispatch: (a: Action) => void }) {
  const nodes = Object.values(state.nodes).sort((a, b) => a.node_id.localeCompare(b.node_id));
  const online = nodes.filter((n) => n.status === "online").length;
  const tracks = Object.values(state.tracks).sort(
    (a, b) => (ORDER[a.status] ?? 9) - (ORDER[b.status] ?? 9) || b.confidence - a.confidence);

  return (
    <aside className="panel left">
      <section>
        <h2 className="caps">Tracks <span className="mono muted">{tracks.length}</span></h2>
        {tracks.length === 0 ? (
          <p className="muted small">No active tracks</p>
        ) : (
          <ul>
            {tracks.map((t) => (
              <TrackRow key={t.track_id} track={t} selected={t.track_id === state.selectedTrackId}
                onSelect={() => dispatch({ type: "select", trackId: t.track_id })} />
            ))}
          </ul>
        )}
      </section>
      <section>
        <h2 className="caps">Sensor nodes <span className="mono muted">{online}/{nodes.length}</span></h2>
        {nodes.length === 0 ? (
          <p className="muted small">No nodes registered</p>
        ) : (
          <ul>{nodes.map((n) => <NodeRow key={n.node_id} node={n} />)}</ul>
        )}
      </section>
    </aside>
  );
}
