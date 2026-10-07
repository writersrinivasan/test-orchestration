// API robustness + live LLM (Groq) evaluation against a running server.
// Run: BASE=http://localhost:3100 npm run test:eval
import { writeFileSync } from "node:fs";
import type { TravelEvent } from "../src/lib/events.ts";
import { SCENARIOS } from "./scenarios.mts";

const BASE = process.env.BASE ?? "http://localhost:3000";
type Turn = { role: "user" | "assistant"; content: string };

async function chat(message: unknown, history: Turn[] = []) {
  const t0 = performance.now();
  const res = await fetch(`${BASE}/api/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: typeof message === "string" && message.startsWith("RAW:") ? message.slice(4) : JSON.stringify({ message, history }),
  });
  const events: TravelEvent[] = [];
  let firstByte = 0, planAt = 0, firstToken = 0, badLines = 0;
  if (res.body) {
    const reader = res.body.getReader();
    const dec = new TextDecoder();
    let buf = "";
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      if (!firstByte) firstByte = performance.now() - t0;
      buf += dec.decode(value, { stream: true });
      const lines = buf.split("\n");
      buf = lines.pop() ?? "";
      for (const l of lines.filter((x) => x.trim())) {
        try {
          const e = JSON.parse(l) as TravelEvent;
          if (e.type === "plan" && !planAt) planAt = performance.now() - t0;
          if (e.type === "token" && !firstToken) firstToken = performance.now() - t0;
          events.push(e);
        } catch {
          badLines++;
        }
      }
    }
  }
  return { status: res.status, events, badLines, ms: performance.now() - t0, firstByte, planAt, firstToken };
}

const pick = <T extends TravelEvent["type"]>(ev: TravelEvent[], t: T) => ev.find((e) => e.type === t) as Extract<TravelEvent, { type: T }> | undefined;
const pct = (xs: number[], p: number) => {
  const s = [...xs].sort((a, b) => a - b);
  return Math.round(s[Math.min(s.length - 1, Math.floor((p / 100) * s.length))] ?? 0);
};
const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

// ---------------- Robustness ----------------
async function robustness() {
  const cases: { name: string; input: unknown; history?: Turn[] }[] = [
    { name: "invalid JSON body", input: "RAW:{not json" },
    { name: "missing message field", input: "RAW:{}" },
    { name: "empty message", input: "" },
    { name: "whitespace only", input: "   " },
    { name: "very long message (20k chars)", input: "Find flights to Paris. ".repeat(900) },
    { name: "HTML/script payload", input: "<script>alert('x')</script> hotels in Goa" },
    { name: "malformed history", input: "hi", history: "oops" as unknown as Turn[] },
  ];
  const rows = [];
  for (const c of cases) {
    const r = await chat(c.input, c.history);
    const final = pick(r.events, "final");
    const err = pick(r.events, "error");
    rows.push({
      name: c.name,
      status: r.status,
      outcome: final ? "answered" : err ? `error event: ${err.message.slice(0, 90)}` : r.status >= 500 ? "HTTP 5xx" : r.status >= 400 ? `rejected ${r.status}` : "no final event",
      // Graceful = clean 4xx rejection, or a normal answer. Crashes inside the graph are not graceful.
      graceful: (r.status >= 400 && r.status < 500) || (r.status === 200 && !!final),
      ms: Math.round(r.ms),
    });
    await sleep(1500);
  }
  return rows;
}

// ---------------- LLM scenarios ----------------
const norm = (v: unknown) => String(v ?? "").toLowerCase().trim();

async function evaluate() {
  const rows = [];
  const only = process.env.ONLY?.split(",");
  for (const s of SCENARIOS.filter((x) => !only || only.includes(x.id))) {
    let r = await chat(s.q, s.history ?? []);
    let retries = 0;
    // Retry once on rate limiting / fallback so results reflect the model, not the quota.
    while (retries < 2 && (pick(r.events, "error")?.message.includes("429") || pick(r.events, "plan")?.reasoning.includes("fallback"))) {
      retries++;
      await sleep(30000);
      r = await chat(s.q, s.history ?? []);
    }
    const plan = pick(r.events, "plan");
    const final = pick(r.events, "final");
    const err = pick(r.events, "error");
    // Compare on rendered text: unicode spaces → space, Markdown bold markers removed.
    const answer = (final?.text ?? "").replace(/[\u00a0\u202f\u2009\u2011]/g, (c) => (c === "\u2011" ? "-" : " ")).replace(/\*\*/g, "");
    const p = plan?.plan ?? [];
    const checks: Record<string, boolean> = {};
    if (s.plan) checks.routingExact = JSON.stringify(p) === JSON.stringify(s.plan);
    if (s.mustInclude || s.mustExclude || s.plan) {
      const inc = (s.mustInclude ?? s.plan ?? []).every((a) => p.includes(a));
      const exc = (s.mustExclude ?? []).every((a) => !p.includes(a));
      checks.routing = inc && exc;
    }
    if (s.params) {
      const ok = Object.entries(s.params).every(([k, v]) => norm((plan?.params as Record<string, unknown>)?.[k]).includes(norm(v)));
      checks.params = ok;
    }
    if (s.mustExclude?.includes("booking")) checks.noUnwantedBooking = !final?.data.booking;
    if (s.mustInclude?.includes("booking") || s.plan?.includes("booking")) {
      const b = final?.data.booking;
      checks.bookingCreated = !!b;
      if (b) checks.bookingMathCorrect = b.total === b.flight.price * b.guests + b.hotel.pricePerNight * b.nights;
    }
    if (s.expectCitation) checks.citation = /[[【]\s*(policies|visa|guides)\//.test(answer);
    if (s.answerHas) checks.answerContent = s.answerHas.every((re) => re.test(answer));
    if (s.answerLacks) checks.answerSafe = s.answerLacks.every((re) => !re.test(answer));
    checks.completed = !!final && !err;
    const passed = Object.values(checks).every(Boolean);
    rows.push({
      id: s.id,
      category: s.category,
      q: s.q,
      plan: p,
      intent: plan?.intent,
      params: plan?.params,
      fallback: !!plan?.reasoning.includes("fallback"),
      retries,
      checks,
      passed,
      ms: Math.round(r.ms),
      planMs: Math.round(r.planAt),
      ttftMs: Math.round(r.firstToken),
      firstByteMs: Math.round(r.firstByte),
      badLines: r.badLines,
      answer: answer.slice(0, 400),
      error: err?.message,
    });
    console.log(`${passed ? "PASS" : "FAIL"} ${s.id} ${s.category} plan=${p.join(",")} ${Math.round(r.ms)}ms`, passed ? "" : JSON.stringify(checks));
    await sleep(4000);
  }
  return rows;
}

// ---------------- Concurrency (live) ----------------
async function concurrency(n: number) {
  const t0 = performance.now();
  const qs = ["What is the baggage policy?", "Hotels in Goa", "Flights Chennai to Dubai", "Visa for Japan?", "Best time to visit Bali?"];
  const res = await Promise.all(Array.from({ length: n }, (_, i) => chat(qs[i % qs.length])));
  return {
    n,
    wallMs: Math.round(performance.now() - t0),
    completed: res.filter((r) => pick(r.events, "final")).length,
    errors: res.filter((r) => pick(r.events, "error")).length,
    fallbacks: res.filter((r) => pick(r.events, "plan")?.reasoning.includes("fallback")).length,
    p50: pct(res.map((r) => r.ms), 50),
    max: Math.round(Math.max(...res.map((r) => r.ms))),
  };
}

const robust = process.env.SKIP_ROBUST ? [] : await robustness();
console.table(robust);
const scen = process.env.SKIP_SCEN ? [] : await evaluate();
const conc = process.env.SKIP_CONC ? null : (await sleep(20000), await concurrency(5));
console.log("concurrency", conc);

const rate = (rs: typeof scen, k: string) => {
  const xs = rs.filter((r) => k in r.checks);
  return { n: xs.length, rate: xs.length ? xs.filter((r) => (r.checks as Record<string, boolean>)[k]).length / xs.length : null };
};
const cats = [...new Set(scen.map((r) => r.category))].map((c) => {
  const xs = scen.filter((r) => r.category === c);
  return { category: c, n: xs.length, passed: xs.filter((r) => r.passed).length };
});
const live = scen.filter((r) => !r.error);
const summary = {
  base: BASE,
  model: process.env.GROQ_MODEL ?? "openai/gpt-oss-120b",
  scenarios: scen.length,
  passed: scen.filter((r) => r.passed).length,
  byCategory: cats,
  checks: Object.fromEntries(
    ["routing", "routingExact", "params", "bookingCreated", "bookingMathCorrect", "noUnwantedBooking", "citation", "answerContent", "answerSafe", "completed"].map((k) => [k, rate(scen, k)]),
  ),
  latencyMs: {
    e2e: { p50: pct(live.map((r) => r.ms), 50), p95: pct(live.map((r) => r.ms), 95), max: pct(live.map((r) => r.ms), 100) },
    plan: { p50: pct(live.map((r) => r.planMs), 50), p95: pct(live.map((r) => r.planMs), 95) },
    ttft: { p50: pct(live.map((r) => r.ttftMs), 50), p95: pct(live.map((r) => r.ttftMs), 95) },
    firstByte: { p50: pct(live.map((r) => r.firstByteMs), 50) },
  },
  fallbacks: scen.filter((r) => r.fallback).length,
  retries: scen.reduce((s, r) => s + r.retries, 0),
  ndjsonBadLines: scen.reduce((s, r) => s + r.badLines, 0),
};
console.log(JSON.stringify(summary, null, 2));
writeFileSync(new URL(`./results/${process.env.OUT ?? "eval"}.json`, import.meta.url), JSON.stringify({ summary, robustness: robust, scenarios: scen, concurrency: conc }, null, 2));
