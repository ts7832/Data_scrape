import { Intent } from "@blueprintjs/core";

// Blueprint's dark-theme palette, for places CSS can't reach (map paint properties).
export const PALETTE = {
  bg: "#1C2127",       // dark-gray1
  text: "#F6F7F9",     // light-gray5
  muted: "#8F99A8",    // gray3
  primary: "#4C90F0",  // blue4
  node: "#68C1EE",     // cerulean4
  warning: "#EC9A3C",  // orange4
  danger: "#E76A6E",   // red4
  success: "#72CA9B",  // green4
} as const;

// Status → Blueprint intent, shared by every table and tag that shows a status.
export const TRACK_INTENT: Record<string, Intent> = {
  confirmed: Intent.DANGER, tentative: Intent.WARNING, downgraded: Intent.NONE, closed: Intent.NONE,
};
export const NODE_INTENT: Record<string, Intent> = {
  online: Intent.SUCCESS, stale: Intent.WARNING, offline: Intent.NONE,
};
