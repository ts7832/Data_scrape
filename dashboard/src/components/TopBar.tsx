import { useEffect, useState } from "react";
import { Alignment, Icon, Intent, Navbar, NavbarDivider, NavbarGroup, NavbarHeading, Tag } from "@blueprintjs/core";
import { clockZ } from "../format";
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
  return <span>{clockZ(now)}</span>;
}

function Stat({ label, value }: { label: string; value: React.ReactNode }) {
  return <span className="stat muted">{label}<b>{value}</b></span>;
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
          <Icon icon="satellite" size={12} /> KUULO
        </NavbarHeading>
        <NavbarDivider />
        <Stat label="SECTOR" value="HELSINKI" />
      </NavbarGroup>
      <NavbarGroup align={Alignment.END}>
        <Tag minimal intent={CONNECTION_INTENT[state.connection]} icon={<Icon icon="full-circle" size={8} />}>
          {state.connection.toUpperCase()}
        </Tag>
        <NavbarDivider />
        <Stat label="SENSORS" value={`${online}/${nodes.length}`} />
        <NavbarDivider />
        <Stat label="TRACKS" value={tracks.length} />
        {confirmed > 0 && <Tag minimal intent={Intent.DANGER}>{confirmed} CONFIRMED</Tag>}
        <NavbarDivider />
        <UtcClock />
      </NavbarGroup>
    </Navbar>
  );
}
