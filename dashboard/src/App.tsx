import { useEffect, useReducer } from "react";
import { fetchSnapshot } from "./api/client";
import { EventLog } from "./components/EventLog";
import { MapView } from "./components/MapView";
import { NodePanel } from "./components/NodePanel";
import { TopBar } from "./components/TopBar";
import { TrackDetail } from "./components/TrackDetail";
import { defaultLiveUrl, startLive, type SocketLike } from "./state/live";
import { initialState, reducer } from "./state/reducer";

export function App() {
  const [state, dispatch] = useReducer(reducer, initialState);

  useEffect(() => startLive({
    url: defaultLiveUrl(),
    // WebSocket's on* handlers take an Event argument SocketLike's don't declare;
    // our handlers never read it, so the runtime shapes match even though the
    // structural (property, not method) check does not see that.
    openSocket: (url) => new WebSocket(url) as unknown as SocketLike,
    fetchSnapshot,
    dispatch,
    schedule: (fn, ms) => { window.setTimeout(fn, ms); },
    now: () => Date.now(),
  }), []);

  useEffect(() => {
    const id = window.setInterval(() => dispatch({ type: "expirePulses", now: Date.now() }), 1000);
    return () => window.clearInterval(id);
  }, []);

  return (
    <div className="layout">
      <TopBar state={state} />
      <NodePanel state={state} dispatch={dispatch} />
      <main className="map-area"><MapView state={state} dispatch={dispatch} /></main>
      <TrackDetail state={state} />
      <EventLog events={state.events} />
    </div>
  );
}
