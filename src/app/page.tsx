"use client";

import { useEffect, useRef, useState } from "react";
import AgentGraph, { edgeKey, type NodeStatus } from "@/components/AgentGraph";
import type { AgentId, FinalData, RagChunk, TravelEvent } from "@/lib/events";

type Msg = {
  role: "user" | "assistant";
  content: string;
  data?: FinalData;
  runId?: string;
  rating?: 0 | 1;
  traceUrl?: string;
};
type Wire = { direction: string; payload: unknown; t: number };
type Rag = Extract<TravelEvent, { type: "rag" }>;
type Plan = Extract<TravelEvent, { type: "plan" }>;
type LogLine = { agent: string; text: string; kind: "log" | "handoff"; t: number };

const SUGGESTIONS = [
  "Find me flights and hotels from Chennai to Tokyo for 4 nights",
  "What is your cancellation policy and baggage allowance?",
  "Do Indian citizens need a visa for Dubai? Best time to visit?",
  "Book a trip from Mumbai to Paris for 2 guests, 5 nights",
];

export default function Home() {
  const [messages, setMessages] = useState<Msg[]>([]);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const [statuses, setStatuses] = useState<Partial<Record<AgentId, NodeStatus>>>({});
  const [traversed, setTraversed] = useState<Set<string>>(new Set());
  const [activeEdge, setActiveEdge] = useState<{ key: string; label: string } | null>(null);
  const [tools, setTools] = useState<{ name: string; description?: string }[]>([]);
  const [wire, setWire] = useState<Wire[]>([]);
  const [rag, setRag] = useState<Rag | null>(null);
  const [plan, setPlan] = useState<Plan | null>(null);
  const [logs, setLogs] = useState<LogLine[]>([]);
  const chatEnd = useRef<HTMLDivElement>(null);

  useEffect(() => {
    chatEnd.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  function handle(e: TravelEvent, t = 0) {
    switch (e.type) {
      case "node":
        setStatuses((s) => ({ ...s, [e.id]: e.status }));
        break;
      case "handoff": {
        const key = edgeKey(e.from, e.to);
        setActiveEdge({ key, label: e.message });
        setTraversed((s) => new Set(s).add(key));
        setLogs((l) => [...l, { agent: `${e.from} → ${e.to}`, text: e.message, kind: "handoff", t }]);
        if (e.to === "supervisor" && e.from === "user") setStatuses((s) => ({ ...s, user: "done" }));
        break;
      }
      case "log":
        setLogs((l) => [...l, { agent: e.agent, text: e.text, kind: "log", t }]);
        break;
      case "plan":
        setPlan(e);
        break;
      case "mcp_tools":
        setTools(e.tools);
        break;
      case "mcp_wire":
        setWire((w) => [...w, { direction: e.direction, payload: e.payload, t }]);
        break;
      case "rag":
        setRag(e);
        break;
      case "token":
        setMessages((m) => {
          const last = m[m.length - 1];
          return [...m.slice(0, -1), { ...last, content: last.content + e.text }];
        });
        break;
      case "final":
        setMessages((m) => [...m.slice(0, -1), { role: "assistant", content: e.text, data: e.data, runId: e.runId }]);
        setActiveEdge(null);
        break;
      case "error":
        setMessages((m) => [...m.slice(0, -1), { role: "assistant", content: `⚠️ ${e.message}` }]);
        break;
    }
  }

  async function send(text: string) {
    if (!text.trim() || busy) return;
    const history = messages.map(({ role, content }) => ({ role, content }));
    setMessages((m) => [...m, { role: "user", content: text }, { role: "assistant", content: "" }]);
    setInput("");
    setBusy(true);
    setStatuses({ user: "active" });
    setTraversed(new Set());
    setActiveEdge(null);
    setWire([]);
    setRag(null);
    setPlan(null);
    setLogs([]);

    try {
      const res = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: text, history }),
      });
      if (!res.ok) {
        const { error } = await res.json().catch(() => ({ error: res.statusText }));
        throw new Error(error);
      }
      const reader = res.body!.getReader();
      const decoder = new TextDecoder();
      let buf = "";
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buf += decoder.decode(value, { stream: true });
        const lines = buf.split("\n");
        buf = lines.pop() ?? "";
        for (const line of lines) if (line.trim()) handle(JSON.parse(line), clock());
      }
      setStatuses((s) => ({ ...s, user: "done" }));
    } catch (err) {
      handle({ type: "error", message: err instanceof Error ? err.message : String(err) });
    } finally {
      setBusy(false);
    }
  }

  return (
    <main className="grid min-h-screen grid-cols-1 text-sm lg:h-screen lg:grid-cols-[400px_1fr] lg:overflow-hidden">
      {/* ---------- Chat ---------- */}
      <section className="flex h-[85vh] min-h-0 flex-col border-b border-slate-800 bg-slate-950 lg:h-auto lg:border-b-0 lg:border-r">
        <header className="border-b border-slate-800 px-4 py-3">
          <h1 className="text-lg font-bold">
            ✈️ SkyWay <span className="text-sky-400">Agentic Travel</span>
          </h1>
          <p className="text-xs text-slate-400">LangGraph multi-agent · MCP tools · RAG · Groq</p>
        </header>

        <div className="scroll-thin flex-1 space-y-3 overflow-y-auto p-4">
          {messages.length === 0 && (
            <div className="space-y-2">
              <p className="text-slate-400">Try one of these:</p>
              {SUGGESTIONS.map((s) => (
                <button
                  key={s}
                  onClick={() => send(s)}
                  className="block w-full rounded-lg border border-slate-800 bg-slate-900 px-3 py-2 text-left text-slate-200 hover:border-sky-500"
                >
                  {s}
                </button>
              ))}
            </div>
          )}
          {messages.map((m, i) => (
            <div key={i} className={m.role === "user" ? "flex justify-end" : ""}>
              <div
                className={`max-w-[92%] whitespace-pre-wrap rounded-2xl px-3 py-2 ${
                  m.role === "user" ? "bg-sky-600 text-white" : "bg-slate-900 text-slate-100"
                }`}
              >
                {m.content ? <Rich text={m.content} /> : <span className="animate-pulse text-slate-400">Agents are working…</span>}
                {m.data && <ResultCards data={m.data} />}
                {m.runId && <Feedback msg={m} onRated={(patch) => setMessages((all) => all.map((x) => (x === m ? { ...x, ...patch } : x)))} />}
              </div>
            </div>
          ))}
          <div ref={chatEnd} />
        </div>

        <form
          onSubmit={(e) => {
            e.preventDefault();
            send(input);
          }}
          className="flex gap-2 border-t border-slate-800 p-3"
        >
          <input
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Ask about trips, flights, hotels, visas…"
            maxLength={2000}
            className="flex-1 rounded-lg border border-slate-700 bg-slate-900 px-3 py-2 outline-none focus:border-sky-500"
          />
          <button disabled={busy} className="rounded-lg bg-sky-600 px-4 font-semibold disabled:opacity-50">
            {busy ? "…" : "Send"}
          </button>
        </form>
      </section>

      {/* ---------- Observability ---------- */}
      <section className="grid min-h-0 grid-rows-[560px_auto] lg:grid-rows-[minmax(0,1.15fr)_minmax(0,1fr)]">
        <div className="relative border-b border-slate-800">
          <PanelTitle title="Agentic AI Workflow" subtitle="LangGraph StateGraph · live" />
          {plan && (
            <div className="absolute left-3 top-12 z-10 max-w-xs rounded-lg border border-slate-700 bg-slate-900/90 p-2 text-xs">
              <div className="font-semibold text-sky-300">Supervisor plan</div>
              <div>
                Intent: <b>{plan.intent}</b>
              </div>
              <div>Route: {["supervisor", ...plan.plan, "responder"].join(" → ")}</div>
              <div className="mt-1 text-slate-400">{plan.reasoning}</div>
              <div className="mt-1 break-all font-mono text-[10px] text-slate-400">{JSON.stringify(plan.params)}</div>
            </div>
          )}
          <AgentGraph statuses={statuses} traversed={traversed} activeEdge={activeEdge} />
          <Legend />
        </div>

        <div className="grid min-h-0 grid-cols-1 divide-y divide-slate-800 md:grid-cols-3 md:divide-x md:divide-y-0">
          <McpPanel tools={tools} wire={wire} />
          <RagPanel rag={rag} />
          <TimelinePanel logs={logs} />
        </div>
      </section>
    </main>
  );
}

function PanelTitle({ title, subtitle }: { title: string; subtitle: string }) {
  return (
    <div className="absolute left-3 top-3 z-10">
      <div className="text-xs font-bold uppercase tracking-wider text-slate-300">{title}</div>
      <div className="text-[10px] text-slate-500">{subtitle}</div>
    </div>
  );
}

function Legend() {
  return (
    <div className="absolute bottom-3 left-3 z-10 flex gap-3 text-[10px] text-slate-400">
      <span><i className="mr-1 inline-block h-2 w-2 rounded-full bg-slate-600" />idle</span>
      <span><i className="mr-1 inline-block h-2 w-2 rounded-full bg-amber-400" />working</span>
      <span><i className="mr-1 inline-block h-2 w-2 rounded-full bg-emerald-500" />done</span>
      <span>dashed = infrastructure layer</span>
    </div>
  );
}

function Panel({ title, subtitle, children }: { title: string; subtitle: string; children: React.ReactNode }) {
  return (
    <div className="flex h-[55vh] min-h-0 flex-col md:h-auto">
      <div className="border-b border-slate-800 px-3 py-2">
        <div className="text-xs font-bold uppercase tracking-wider text-slate-300">{title}</div>
        <div className="text-[10px] text-slate-500">{subtitle}</div>
      </div>
      <div className="scroll-thin flex-1 space-y-2 overflow-y-auto p-3">{children}</div>
    </div>
  );
}

function McpPanel({ tools, wire }: { tools: { name: string; description?: string }[]; wire: Wire[] }) {
  return (
    <Panel title="🔌 MCP Layer" subtitle="Client ⇄ travel-tools server · JSON-RPC 2.0">
      <div className="flex flex-wrap gap-1">
        {tools.length === 0 && <span className="text-slate-500">Tools appear after tools/list…</span>}
        {tools.map((t) => (
          <span key={t.name} title={t.description} className="rounded bg-violet-500/15 px-2 py-0.5 font-mono text-[11px] text-violet-300">
            {t.name}
          </span>
        ))}
      </div>
      {wire.map((w, i) => {
        const p = w.payload as { method?: string; id?: number; params?: unknown; result?: unknown };
        const out = w.direction === "client→server";
        const label = p.method ?? `response #${p.id}`;
        return (
          <details key={i} className={`rounded border px-2 py-1 ${out ? "border-sky-800 bg-sky-950/40" : "border-violet-800 bg-violet-950/40"}`}>
            <summary className="cursor-pointer font-mono text-[11px]">
              <span className={out ? "text-sky-300" : "text-violet-300"}>{out ? "→ client" : "← server"}</span>{" "}
              {label}
              {p.method === "tools/call" && (
                <span className="text-amber-300"> {(p.params as { name: string }).name}</span>
              )}
            </summary>
            <pre className="mt-1 max-h-40 overflow-auto whitespace-pre-wrap break-all font-mono text-[10px] text-slate-400">
              {JSON.stringify(w.payload, null, 2)}
            </pre>
          </details>
        );
      })}
    </Panel>
  );
}

function RagPanel({ rag }: { rag: Rag | null }) {
  const steps = ["Query", "Embed", "Vector search", "Top-k chunks", "Augment prompt"];
  return (
    <Panel title="📚 RAG Layer" subtitle="Retrieve → Augment → Generate">
      <div className="flex flex-wrap items-center gap-1 text-[10px]">
        {steps.map((s, i) => (
          <span key={s} className="flex items-center gap-1">
            <span className={`rounded px-1.5 py-0.5 ${rag ? "bg-emerald-500/20 text-emerald-300" : "bg-slate-800 text-slate-500"}`}>{s}</span>
            {i < steps.length - 1 && <span className="text-slate-600">→</span>}
          </span>
        ))}
      </div>
      {!rag && <p className="text-slate-500">Ask a policy, visa or destination question to see retrieval.</p>}
      {rag && (
        <>
          <div className="rounded bg-slate-900 p-2 text-xs">
            <div className="text-slate-400">Query</div>
            <div className="text-slate-100">“{rag.query}”</div>
            <div className="mt-1 text-slate-400">Embedding (non-zero dims of 256)</div>
            <div className="font-mono text-[10px] text-emerald-300">[{rag.queryVector.join(", ")}…]</div>
            <div className="mt-1 text-[10px] text-slate-500">
              Searched {rag.totalChunks} chunks in {rag.ms} ms · cosine similarity
            </div>
          </div>
          {rag.chunks.map((c: RagChunk, i) => (
            <div key={c.id} className="rounded border border-slate-800 bg-slate-900/60 p-2">
              <div className="flex items-center justify-between text-[11px]">
                <span className="font-semibold text-slate-200">#{i + 1} {c.title}</span>
                <span className="font-mono text-emerald-300">{c.score.toFixed(3)}</span>
              </div>
              <div className="my-1 h-1 rounded bg-slate-800">
                <div className="h-1 rounded bg-emerald-500" style={{ width: `${Math.min(100, c.score * 100)}%` }} />
              </div>
              <div className="font-mono text-[10px] text-slate-500">{c.id}</div>
              <div className="mt-1 text-[11px] text-slate-400">{c.text}</div>
            </div>
          ))}
        </>
      )}
    </Panel>
  );
}

function TimelinePanel({ logs }: { logs: LogLine[] }) {
  const start = logs[0]?.t ?? 0;
  return (
    <Panel title="🧵 Agent Messages" subtitle="Handoffs & reasoning over shared state">
      {logs.length === 0 && <p className="text-slate-500">Agent-to-agent communication appears here.</p>}
      {logs.map((l, i) => (
        <div key={i} className="flex gap-2 text-[11px]">
          <span className="w-10 shrink-0 font-mono text-slate-600">+{((l.t - start) / 1000).toFixed(1)}s</span>
          <div>
            <span className={l.kind === "handoff" ? "font-semibold text-amber-300" : "font-semibold text-sky-300"}>
              {l.kind === "handoff" ? "⇢ " : ""}
              {l.agent}
            </span>
            <span className="text-slate-300"> {l.text}</span>
          </div>
        </div>
      ))}
    </Panel>
  );
}

// Sends the customer's rating to LangSmith as feedback on this answer's trace.
function Feedback({ msg, onRated }: { msg: Msg; onRated: (patch: Partial<Msg>) => void }) {
  const [error, setError] = useState("");
  async function rate(score: 0 | 1) {
    setError("");
    const res = await fetch("/api/feedback", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ runId: msg.runId, score }),
    });
    const body = await res.json().catch(() => ({}));
    if (!res.ok) return setError(body.error ?? "Couldn't save feedback. Try again.");
    onRated({ rating: score, traceUrl: body.traceUrl });
  }
  return (
    <div className="mt-2 flex items-center gap-2 whitespace-normal border-t border-slate-800 pt-2 text-[11px] text-slate-400">
      {msg.rating === undefined ? (
        <>
          <span>Was this helpful?</span>
          <button onClick={() => rate(1)} aria-label="Helpful" className="rounded px-1.5 hover:bg-slate-800">👍</button>
          <button onClick={() => rate(0)} aria-label="Not helpful" className="rounded px-1.5 hover:bg-slate-800">👎</button>
        </>
      ) : (
        <span>{msg.rating ? "👍" : "👎"} Saved to LangSmith</span>
      )}
      {msg.traceUrl && (
        <a href={msg.traceUrl} target="_blank" rel="noreferrer" className="text-sky-400 underline">
          Open trace
        </a>
      )}
      {error && <span className="text-rose-400">{error}</span>}
    </div>
  );
}

function ResultCards({ data }: { data: FinalData }) {
  return (
    <div className="mt-2 space-y-2 whitespace-normal">
      {data.booking && (
        <div className="rounded-lg border border-emerald-600 bg-emerald-950/50 p-2">
          <div className="font-bold text-emerald-300">✅ Booking {data.booking.status}</div>
          <div className="font-mono text-lg">{data.booking.reference}</div>
          <div className="text-xs text-slate-300">
            {data.booking.flight.airline} {data.booking.flight.from} → {data.booking.flight.to} · {data.booking.hotel.name} ·{" "}
            {data.booking.nights} nights · {data.booking.guests} guest(s)
          </div>
          <div className="font-semibold">Total ${data.booking.total}</div>
        </div>
      )}
      {!!data.flights?.length && (
        <div>
          <div className="mb-1 text-xs font-semibold text-slate-400">✈️ Flights</div>
          {data.flights.map((f) => (
            <div key={f.id} className="mb-1 flex justify-between rounded bg-slate-800 px-2 py-1 text-xs">
              <span>
                {f.airline} · {f.depart}→{f.arrive} · {f.stops ? `${f.stops} stop` : "nonstop"}
              </span>
              <span className="font-semibold text-sky-300">${f.price}</span>
            </div>
          ))}
        </div>
      )}
      {!!data.hotels?.length && (
        <div>
          <div className="mb-1 text-xs font-semibold text-slate-400">🏨 Hotels</div>
          {data.hotels.map((h) => (
            <div key={h.id} className="mb-1 flex justify-between rounded bg-slate-800 px-2 py-1 text-xs">
              <span>
                {h.name} · {"★".repeat(h.stars)} · {h.rating}/5
              </span>
              <span className="font-semibold text-sky-300">${h.pricePerNight}/n</span>
            </div>
          ))}
        </div>
      )}
      {data.weather && (
        <div className="text-xs text-slate-400">
          🌤️ {String(data.weather.city)} on {String(data.weather.date)}: {String(data.weather.condition)}, {String(data.weather.highC)}°/
          {String(data.weather.lowC)}°C
        </div>
      )}
    </div>
  );
}

// Wall-clock time for the timeline; only ever called from the streaming loop, never during render.
function clock() {
  return Date.now();
}

// Minimal Markdown: **bold** and leading "- " bullets, enough for the assistant's replies.
function Rich({ text }: { text: string }) {
  return (
    <>
      {text
        .replace(/^(\s*)[-*] /gm, "$1• ")
        .split(/(\*\*[^*]+\*\*)/g)
        .map((part, i) =>
          part.startsWith("**") && part.endsWith("**") ? <strong key={i}>{part.slice(2, -2)}</strong> : part,
        )}
    </>
  );
}
