import { travelGraph, type ChatTurn } from "@/lib/graph";
import { openMcpSession } from "@/lib/mcp";
import type { TravelEvent } from "@/lib/events";
import { awaitAllCallbacks } from "@langchain/core/callbacks/promises";

const MAX_MESSAGE = 2000;
const MAX_HISTORY = 10;

// Streams newline-delimited JSON events: every agent step, MCP message and RAG lookup.
export async function POST(req: Request) {
  let body: { message?: unknown; history?: unknown };
  try {
    body = await req.json();
  } catch {
    return Response.json({ error: "Body must be JSON: { message, history? }" }, { status: 400 });
  }
  const message = typeof body.message === "string" ? body.message.trim() : "";
  if (!message) return Response.json({ error: "message must be a non-empty string" }, { status: 400 });
  if (message.length > MAX_MESSAGE) {
    return Response.json({ error: `message exceeds ${MAX_MESSAGE} characters` }, { status: 413 });
  }
  // Keep only well-formed recent turns; bounds prompt size and LLM cost.
  const history: ChatTurn[] = (Array.isArray(body.history) ? body.history : [])
    .filter((t): t is ChatTurn => (t?.role === "user" || t?.role === "assistant") && typeof t?.content === "string")
    .slice(-MAX_HISTORY)
    .map((t) => ({ role: t.role, content: t.content.slice(0, MAX_MESSAGE) }));
  const encoder = new TextEncoder();

  const stream = new ReadableStream({
    async start(controller) {
      const emit = (e: TravelEvent) => controller.enqueue(encoder.encode(JSON.stringify(e) + "\n"));
      const mcp = await openMcpSession(emit);
      // The run id ties this request's LangSmith trace to the feedback the user gives on the answer.
      const runId = crypto.randomUUID();
      try {
        const result = await travelGraph.invoke(
          { query: message, history },
          {
            configurable: { emit, mcp },
            recursionLimit: 25,
            runId,
            runName: "skyway-chat",
            tags: ["chat", process.env.NODE_ENV ?? "development"],
            metadata: { model: process.env.GROQ_MODEL || "openai/gpt-oss-120b", historyTurns: history.length },
          },
        );
        emit({
          type: "final",
          text: result.answer,
          data: {
            flights: result.flights.slice(0, 3),
            hotels: result.hotels.slice(0, 3),
            booking: result.confirmation ?? undefined,
            weather: result.weather ?? undefined,
          },
          runId: process.env.LANGSMITH_TRACING === "true" ? runId : undefined,
        });
      } catch (err) {
        console.error(err);
        emit({ type: "error", message: err instanceof Error ? err.message : String(err) });
      } finally {
        await mcp.close();
        controller.close();
        await awaitAllCallbacks(); // flush LangSmith traces after the user has their answer
      }
    },
  });

  return new Response(stream, { headers: { "Content-Type": "application/x-ndjson; charset=utf-8", "Cache-Control": "no-cache" } });
}
