# SkyWay Agentic Travel

A Next.js travel booking assistant built with **LangGraph** multi-agent orchestration, a real **MCP** server and client, a **RAG** knowledge base, and **Groq** as the LLM. As each question is answered, the UI shows the agent workflow, the MCP traffic and the RAG retrieval live.

## Run

```bash
cp .env.example .env.local   # then paste your key: GROQ_API_KEY=gsk_...
npm install
npm run dev                  # http://localhost:3000
```

Without a key, the app runs in **demo mode**: it routes by keyword and answers from a template, so the graph, MCP and RAG panels still work.

## Architecture

```
Customer ─► Supervisor (Groq: intent + plan + params)
               │  hub & spoke: each specialist hands control back to the supervisor
               ├─► Research Agent ──► Vector Store   (RAG: TF-IDF embeddings + cosine top-k)
               ├─► Flight Agent  ──┐
               ├─► Hotel Agent   ──┼─► MCP Server "travel-tools" (JSON-RPC 2.0)
               ├─► Booking Agent ──┘   search_flights · search_hotels · get_weather · create_booking
               └─► Responder (Groq, streams tokens, cites RAG sources) ─► Customer
```

| File | What it shows |
| --- | --- |
| `src/lib/graph.ts` | LangGraph `StateGraph`: shared state, supervisor routing, conditional edges |
| `src/lib/mcp.ts` | MCP server and client using `@modelcontextprotocol/sdk` over an in-memory transport; every JSON-RPC message is mirrored to the UI |
| `src/lib/rag.ts` | Documents → chunks → embeddings → cosine search |
| `src/app/api/chat/route.ts` | Streams NDJSON events (node status, handoffs, MCP wire, RAG hits, tokens) |
| `src/app/page.tsx`, `src/components/AgentGraph.tsx` | Chat, live React Flow graph, MCP/RAG/agent message panels |

## Demo script

1. **RAG:** "What is your cancellation policy and baggage allowance?" Only the Research Agent runs. Point out the query embedding, the similarity scores and the `[source]` citations.
2. **MCP:** "Find me flights and hotels from Chennai to Tokyo for 4 nights". Open the MCP panel and walk through `initialize` → `tools/list` → `tools/call`.
3. **Full orchestration:** "Book a trip from Mumbai to Paris for 2 guests, 5 nights". All agents run, and `create_booking` returns a `SKY…` reference.
4. **Memory:** follow up with "Actually make it 3 nights and book it". The supervisor reuses details from earlier turns.
