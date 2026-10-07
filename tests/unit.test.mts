// Unit, contract and offline-orchestration tests. Run: npm run test:unit
// Runs with GROQ_API_KEY unset so the graph uses keyword routing (deterministic, free).
delete process.env.GROQ_API_KEY;

import { test, describe } from "node:test";
import assert from "node:assert/strict";
import { writeFileSync } from "node:fs";
import { retrieve } from "../src/lib/rag.ts";
import { openMcpSession } from "../src/lib/mcp.ts";
import { travelGraph } from "../src/lib/graph.ts";
import type { Booking, Flight, Hotel, TravelEvent } from "../src/lib/events.ts";

const metrics: Record<string, unknown> = {};

// ---------------- RAG ----------------
// Labelled queries → the source document a correct retrieval must return.
const RAG_GOLD: { q: string; expect: string; kind: "keyword" | "paraphrase" }[] = [
  { q: "What is the cancellation policy?", expect: "policies/cancellation.md", kind: "keyword" },
  { q: "How much is the hotel cancellation fee?", expect: "policies/cancellation.md", kind: "keyword" },
  { q: "How many checked bags are included?", expect: "policies/baggage.md", kind: "keyword" },
  { q: "Can I bring my surfboard?", expect: "policies/baggage.md", kind: "keyword" },
  { q: "Which payment methods do you accept? UPI?", expect: "policies/payments.md", kind: "keyword" },
  { q: "Can I hold a fare with price lock?", expect: "policies/payments.md", kind: "keyword" },
  { q: "What does travel insurance cover?", expect: "policies/insurance.md", kind: "keyword" },
  { q: "How do loyalty points and Gold status work?", expect: "policies/loyalty.md", kind: "keyword" },
  { q: "Do Indian citizens need a visa for Japan?", expect: "visa/japan.md", kind: "keyword" },
  { q: "Visa on arrival in Dubai UAE", expect: "visa/uae.md", kind: "keyword" },
  { q: "Schengen visa for France ETIAS", expect: "visa/schengen.md", kind: "keyword" },
  { q: "Best time to visit Paris and the Louvre", expect: "guides/paris.md", kind: "keyword" },
  { q: "Cherry blossom season in Tokyo", expect: "guides/tokyo.md", kind: "keyword" },
  { q: "Surfing and rice terraces in Bali", expect: "guides/bali.md", kind: "keyword" },
  { q: "Desert safari and Burj Khalifa", expect: "guides/dubai.md", kind: "keyword" },
  { q: "Broadway and Central Park in New York", expect: "guides/new-york.md", kind: "keyword" },
  { q: "Goa beaches nightlife peak season", expect: "guides/goa.md", kind: "keyword" },
  // Paraphrases with little word overlap: probes the limits of TF-IDF retrieval.
  { q: "Can I get my money back if I change my mind?", expect: "policies/cancellation.md", kind: "paraphrase" },
  { q: "How heavy can my suitcase be?", expect: "policies/baggage.md", kind: "paraphrase" },
  { q: "What happens if I get sick abroad?", expect: "policies/insurance.md", kind: "paraphrase" },
  { q: "Do I need any paperwork to enter Tokyo?", expect: "visa/japan.md", kind: "paraphrase" },
  { q: "When is it not too hot in the Emirates?", expect: "guides/dubai.md", kind: "paraphrase" },
];

describe("RAG retrieval", () => {
  test("returns k results sorted by score, with valid shape", () => {
    const r = retrieve("baggage allowance", 3);
    assert.equal(r.chunks.length, 3);
    for (let i = 1; i < r.chunks.length; i++) assert.ok(r.chunks[i - 1].score >= r.chunks[i].score);
    for (const c of r.chunks) {
      assert.ok(c.score >= 0 && c.score <= 1.0001);
      assert.match(c.id, /#\d+$/);
    }
    assert.ok(r.totalChunks >= 14);
  });

  test("empty / nonsense query does not throw", () => {
    assert.doesNotThrow(() => retrieve("", 3));
    assert.doesNotThrow(() => retrieve("zzqx ##!! 12345", 3));
  });

  test("retrieval quality on labelled set (Hit@1, Hit@3, MRR)", () => {
    const rows = RAG_GOLD.map((g) => {
      const t0 = performance.now();
      const r = retrieve(g.q, 3);
      const ms = performance.now() - t0;
      const rank = r.chunks.findIndex((c) => c.source === g.expect) + 1; // 0 = miss
      return { ...g, rank, top: r.chunks[0].source, topScore: r.chunks[0].score, ms };
    });
    const agg = (subset: typeof rows) => ({
      n: subset.length,
      hit1: subset.filter((r) => r.rank === 1).length / subset.length,
      hit3: subset.filter((r) => r.rank >= 1).length / subset.length,
      mrr: subset.reduce((s, r) => s + (r.rank ? 1 / r.rank : 0), 0) / subset.length,
    });
    const lat = rows.map((r) => r.ms).sort((a, b) => a - b);
    metrics.rag = {
      overall: agg(rows),
      keyword: agg(rows.filter((r) => r.kind === "keyword")),
      paraphrase: agg(rows.filter((r) => r.kind === "paraphrase")),
      latencyMs: { p50: lat[Math.floor(lat.length / 2)], max: lat[lat.length - 1] },
      misses: rows.filter((r) => r.rank !== 1).map((r) => ({ q: r.q, expect: r.expect, got: r.top, rank: r.rank })),
    };
    // Keyword-style questions must be answered reliably.
    assert.ok((metrics.rag as { keyword: { hit3: number } }).keyword.hit3 >= 0.9);
  });
});

// ---------------- MCP ----------------
describe("MCP server/client contract", () => {
  test("handshake, tools/list and all four tools", async () => {
    const wire: TravelEvent[] = [];
    const mcp = await openMcpSession((e) => wire.push(e));
    try {
      const methods = wire.filter((e) => e.type === "mcp_wire").map((e) => (e as { payload: { method?: string } }).payload.method);
      assert.deepEqual(methods.slice(0, 2), ["initialize", undefined]); // request, then response
      assert.ok(methods.includes("notifications/initialized"));
      assert.ok(methods.includes("tools/list"));
      const tools = wire.find((e) => e.type === "mcp_tools") as { tools: { name: string }[] };
      assert.deepEqual(tools.tools.map((t) => t.name).sort(), ["create_booking", "get_weather", "search_flights", "search_hotels"]);
      for (const m of wire.filter((e) => e.type === "mcp_wire")) assert.equal((m as { payload: { jsonrpc: string } }).payload.jsonrpc, "2.0");

      const f1 = await mcp.call<Flight[]>("search_flights", { from: "Chennai", to: "Tokyo", date: "2026-11-01" });
      const f2 = await mcp.call<Flight[]>("search_flights", { from: "Chennai", to: "Tokyo", date: "2026-11-01" });
      assert.equal(f1.length, 5);
      assert.deepEqual(f1, f2, "search is deterministic");
      for (let i = 1; i < f1.length; i++) assert.ok(f1[i - 1].price <= f1[i].price, "sorted by price");
      for (const f of f1) assert.ok(f.price > 0 && /^\d\d:\d\d$/.test(f.depart) && /^\d\d:\d\d$/.test(f.arrive));

      const h = await mcp.call<Hotel[]>("search_hotels", { city: "Tokyo" });
      assert.ok(h.length > 0);
      for (let i = 1; i < h.length; i++) assert.ok(h[i - 1].rating >= h[i].rating, "sorted by rating");
      const cheap = await mcp.call<Hotel[]>("search_hotels", { city: "Tokyo", maxPrice: 200 });
      assert.ok(cheap.every((x) => x.pricePerNight <= 200), "maxPrice filter honoured");

      const w = await mcp.call<{ highC: number; lowC: number }>("get_weather", { city: "Tokyo", date: "2026-11-01" });
      assert.ok(typeof w.highC === "number");

      const b = await mcp.call<Booking>("create_booking", { flight: f1[0], hotel: h[0], nights: 4, guests: 2 });
      assert.match(b.reference, /^SKY[A-Z0-9]{6}$/);
      assert.equal(b.status, "CONFIRMED");
      assert.equal(b.total, f1[0].price * 2 + h[0].pricePerNight * 4, "total = flight×guests + hotel×nights");
    } finally {
      await mcp.close();
    }
  });

  test("schema validation rejects bad arguments", async () => {
    const mcp = await openMcpSession(() => {});
    try {
      // Errors must surface as readable MCP errors, not JSON parse failures.
      await assert.rejects(mcp.call("search_flights", { from: "Chennai" }), /MCP tool search_flights failed/);
      await assert.rejects(mcp.call("search_hotels", { city: "Paris", maxPrice: "cheap" }), /MCP tool search_hotels failed/);
      await assert.rejects(mcp.call("no_such_tool", {}), /no_such_tool/);
    } finally {
      await mcp.close();
    }
  });

  test("MCP session overhead", async () => {
    const times: number[] = [];
    for (let i = 0; i < 20; i++) {
      const t0 = performance.now();
      const s = await openMcpSession(() => {});
      await s.call("search_flights", { from: "A", to: "B", date: "2026-11-01" });
      await s.close();
      times.push(performance.now() - t0);
    }
    times.sort((a, b) => a - b);
    metrics.mcp = { sessionPlusCallMs: { p50: times[10], p95: times[18], max: times[19] } };
  });
});

// ---------------- Orchestration (offline, keyword routing) ----------------
async function run(query: string, history: { role: "user" | "assistant"; content: string }[] = []) {
  const events: TravelEvent[] = [];
  const emit = (e: TravelEvent) => events.push(e);
  const mcp = await openMcpSession(emit);
  try {
    const result = await travelGraph.invoke({ query, history }, { configurable: { emit, mcp }, recursionLimit: 25 });
    return { result, events };
  } finally {
    await mcp.close();
  }
}

const ROUTING: { q: string; plan: string[] }[] = [
  { q: "What is your baggage policy?", plan: ["research"] },
  { q: "Find flights from Delhi to Dubai", plan: ["flight"] },
  { q: "Find a hotel in Goa", plan: ["hotel"] },
  { q: "Plan a trip to Bali", plan: ["flight", "hotel"] },
  { q: "Book a trip from Mumbai to Paris for 2 guests", plan: ["flight", "hotel", "booking"] },
  { q: "hello", plan: ["research"] },
];

describe("LangGraph orchestration (offline)", () => {
  for (const c of ROUTING) {
    test(`routes "${c.q}" → ${c.plan.join(",")}`, async () => {
      const { events } = await run(c.q);
      const plan = events.find((e) => e.type === "plan") as { plan: string[] };
      assert.deepEqual(plan.plan, c.plan);
    });
  }

  test("event protocol invariants on a full booking run", async () => {
    const t0 = performance.now();
    const { result, events } = await run("Book a trip from Mumbai to Paris for 2 guests, 5 nights. Cancellation policy?");
    const ms = performance.now() - t0;
    // every node that starts also finishes, in order
    const nodeEv = events.filter((e) => e.type === "node") as { id: string; status: string }[];
    const open = new Map<string, number>();
    for (const n of nodeEv) open.set(n.id, (open.get(n.id) ?? 0) + (n.status === "active" ? 1 : -1));
    for (const [id, v] of open) assert.equal(v, 0, `${id} active/done balanced`);
    // specialists visited in canonical order, supervisor between each
    const visits = nodeEv.filter((n) => n.status === "active" && !["mcp", "vector"].includes(n.id)).map((n) => n.id);
    assert.deepEqual(visits, ["supervisor", "research", "supervisor", "flight", "supervisor", "hotel", "supervisor", "booking", "supervisor", "responder"]);
    // booking used results from upstream agents
    assert.ok(result.confirmation);
    assert.equal(result.confirmation!.flight.id, result.flights[0].id);
    assert.equal(result.confirmation!.hotel.id, result.hotels[0].id);
    assert.equal(result.confirmation!.guests, 2);
    assert.equal(result.confirmation!.nights, 5);
    assert.ok(events.some((e) => e.type === "rag"));
    metrics.offlineFullRunMs = Math.round(ms);
  });

  test("unknown destination still completes (graceful)", async () => {
    const { result } = await run("Find flights to Atlantis");
    assert.ok(result.answer.length > 0);
  });
});

test.after(() => {
  writeFileSync(new URL("./results/unit-metrics.json", import.meta.url), JSON.stringify(metrics, null, 2));
});
