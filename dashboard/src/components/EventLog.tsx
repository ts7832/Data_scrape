import { HTMLTable, Section, SectionCard, Tag } from "@blueprintjs/core";
import { clockZ } from "../format";
import type { LogEntry } from "../state/reducer";

export function EventLog({ events }: { events: LogEntry[] }) {
  return (
    <section className="panel log">
      <Section compact title="Event log" rightElement={<Tag minimal>{events.length}</Tag>}>
        <SectionCard padded={false}>
          <HTMLTable compact striped>
            <colgroup><col style={{ width: 88 }} /><col style={{ width: 96 }} /><col /></colgroup>
            <thead><tr><th>TIME</th><th>SRC</th><th>EVENT</th></tr></thead>
            <tbody>
              {events.length === 0 ? (
                <tr className="empty"><td colSpan={3}>NO EVENTS</td></tr>
              ) : events.map((e) => (
                <tr key={e.seq}>
                  <td className="muted">{clockZ(e.at)}</td>
                  <td>{e.source}</td>
                  <td className={`tone-${e.tone}`}>{e.message}</td>
                </tr>
              ))}
            </tbody>
          </HTMLTable>
        </SectionCard>
      </Section>
    </section>
  );
}
