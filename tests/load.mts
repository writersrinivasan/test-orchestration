// Offline load test: runs the LangGraph + MCP + RAG pipeline in-process with keyword
// routing (no Groq), so it measures orchestration capacity, not LLM quota.
// Run: npm run test:load
delete process.env.GROQ_API_KEY;
import { writeFileSync } from "node:fs";
import { travelGraph } from "../src/lib/graph.ts";
import { openMcpSession } from "../src/lib/mcp.ts";
import type { TravelEvent } from "../src/lib/events.ts";

async function once(q: string) {
  const t0 = performance.now();
  const events: TravelEvent[] = [];
  const emit = (e: TravelEvent) => events.push(e);
  const mcp = await openMcpSession(emit);
  try {
    const r = await travelGraph.invoke({ query: q, history: [] }, { configurable: { emit, mcp }, recursionLimit: 25 });
    return { ok: !!r.answer, ms: performance.now() - t0, events: events.length };
  } catch {
    return { ok: false, ms: performance.now() - t0, events: events.length };
  } finally {
    await mcp.close();
  }
}

const Q = "Book a trip from Mumbai to Paris for 2 guests, 5 nights. Cancellation policy?"; // heaviest path: 4 agents
const pct = (xs: number[], p: number) => Math.round([...xs].sort((a, b) => a - b)[Math.min(xs.length - 1, Math.floor((p / 100) * xs.length))]);
const rows = [];
for (const n of [1, 10, 25, 50, 100]) {
  const rss0 = process.memoryUsage().rss;
  const t0 = performance.now();
  const res = await Promise.all(Array.from({ length: n }, () => once(Q)));
  const wall = performance.now() - t0;
  const ms = res.map((r) => r.ms);
  rows.push({
    concurrency: n,
    success: `${res.filter((r) => r.ok).length}/${n}`,
    wallMs: Math.round(wall),
    p50Ms: pct(ms, 50),
    p95Ms: pct(ms, 95),
    throughputPerMin: Math.round((n / wall) * 60000),
    eventsPerRun: res[0].events,
    rssDeltaMB: Math.round((process.memoryUsage().rss - rss0) / 1048576),
  });
  console.log(rows.at(-1));
}
// The UI-pacing delays in graph.ts (300–500 ms per step) are intentional; ~3.3 s of each run is that pacing.
writeFileSync(new URL("./results/load.json", import.meta.url), JSON.stringify(rows, null, 2));
