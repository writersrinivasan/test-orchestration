import { Client } from "langsmith";

const UUID = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;

// Records a customer's 👍/👎 on an answer as LangSmith feedback on that answer's trace.
export async function POST(req: Request) {
  if (!process.env.LANGSMITH_API_KEY) return Response.json({ error: "LangSmith is not configured" }, { status: 503 });
  const body = (await req.json().catch(() => ({}))) as { runId?: unknown; score?: unknown; comment?: unknown };
  if (typeof body.runId !== "string" || !UUID.test(body.runId)) {
    return Response.json({ error: "runId must be a UUID" }, { status: 400 });
  }
  if (body.score !== 0 && body.score !== 1) return Response.json({ error: "score must be 0 or 1" }, { status: 400 });

  const client = new Client();
  const feedback = await client.createFeedback(body.runId, "user_rating", {
    score: body.score,
    comment: typeof body.comment === "string" ? body.comment.slice(0, 500) : undefined,
    feedbackSourceType: "app",
  });
  let traceUrl: string | undefined;
  try {
    traceUrl = await client.getRunUrl({ runId: body.runId });
  } catch {
    // The trace may still be ingesting; the feedback is saved either way.
  }
  return Response.json({ ok: true, feedbackId: feedback.id, traceUrl });
}
