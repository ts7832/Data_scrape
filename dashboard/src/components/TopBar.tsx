import { useEffect, useState } from "react";
import { Alignment, Icon, Intent, Navbar, NavbarDivider, NavbarGroup, NavbarHeading, Tag } from "@blueprintjs/core";
import { clockUtc } from "../format";
import type { Connection, State } from "../state/reducer";

const CONNECTION_INTENT: Record<Connection, Intent> = {
  live: Intent.SUCCESS, reconnecting: Intent.WARNING, connecting: Intent.NONE,
};

function UtcClock() {
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    const id = window.setInterval(() => setNow(Date.now()), 1000);
    return () => window.clearInterval(id);
  }, []);
  return <span className="mono">UTC {clockUtc(now)}</span>;
}

export function TopBar({ state }: { state: State }) {
  const nodes = Object.values(state.nodes);
  const online = nodes.filter((n) => n.status === "online").length;
  const tracks = Object.values(state.tracks);
  const confirmed = tracks.filter((t) => t.status === "confirmed").length;

  return (
    <Navbar className="topbar">
      <NavbarGroup align={Alignment.START}>
        <NavbarHeading className="brand">
          <Icon icon="satellite" size={14} /> KUULO
        </NavbarHeading>
        <NavbarDivider />
        <span className="caps muted">Helsinki sector</span>
      </NavbarGroup>
      <NavbarGroup align={Alignment.END}>
        <Tag minimal round intent={CONNECTION_INTENT[state.connection]} icon="dot">
          {state.connection.toUpperCase()}
        </Tag>
        <NavbarDivider />
        <span className="caps muted">Nodes</span>&nbsp;<span className="mono">{online}/{nodes.length}</span>
        <NavbarDivider />
        <span className="caps muted">Tracks</span>&nbsp;<span className="mono">{tracks.length}</span>
        {confirmed > 0 && (
          <Tag minimal intent={Intent.DANGER} className="gap-left">{confirmed} CONFIRMED</Tag>
        )}
        <NavbarDivider />
        <UtcClock />
      </NavbarGroup>
    </Navbar>
  );
}
