import { describe, expect, it } from "vitest";
import { clockUtc, clockZ, isoZ, shortTrackId } from "./format";

describe("format", () => {
  it("shortens a track uuid to a readable callsign", () => {
    expect(shortTrackId("3f2a9c10-1111-4222-8333-444455556666")).toBe("T-3F2A");
  });

  it("formats a UTC clock as HH:MM:SS", () => {
    expect(clockUtc(Date.UTC(2026, 8, 24, 9, 4, 7))).toBe("09:04:07");
  });

  it("formats a Zulu time as HH:MM:SSZ", () => {
    expect(clockZ(Date.UTC(2026, 8, 24, 9, 4, 7))).toBe("09:04:07Z");
    expect(isoZ("2026-09-24T09:04:07+00:00")).toBe("09:04:07Z");
  });
});
