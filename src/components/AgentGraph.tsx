"use client";

import { useMemo } from "react";
import {
  ReactFlow,
  Background,
  Handle,
  Position,
  ConnectionMode,
  MarkerType,
  type Edge,
  type Node,
  type NodeProps,
} from "@xyflow/react";
import "@xyflow/react/dist/style.css";
import type { AgentId } from "@/lib/events";

export type NodeStatus = "idle" | "active" | "done";

type AgentNodeData = { label: string; sub: string; icon: string; status: NodeStatus; kind: "agent" | "infra" | "user" };

const NODES: { id: AgentId; x: number; y: number; label: string; sub: string; icon: string; kind: AgentNodeData["kind"] }[] = [
  { id: "user", x: 0, y: 200, label: "Customer", sub: "chat UI", icon: "🧑", kind: "user" },
  { id: "supervisor", x: 200, y: 200, label: "Supervisor", sub: "plans & routes (Groq)", icon: "🧠", kind: "agent" },
  { id: "research", x: 450, y: 0, label: "Research Agent", sub: "RAG knowledge", icon: "📚", kind: "agent" },
  { id: "flight", x: 450, y: 130, label: "Flight Agent", sub: "search_flights", icon: "✈️", kind: "agent" },
  { id: "hotel", x: 450, y: 260, label: "Hotel Agent", sub: "search_hotels · weather", icon: "🏨", kind: "agent" },
  { id: "booking", x: 450, y: 390, label: "Booking Agent", sub: "create_booking", icon: "🎫", kind: "agent" },
  { id: "vector", x: 710, y: 0, label: "Vector Store", sub: "RAG layer", icon: "🧮", kind: "infra" },
  { id: "mcp", x: 710, y: 245, label: "MCP Server", sub: "travel-tools", icon: "🔌", kind: "infra" },
  { id: "responder", x: 200, y: 390, label: "Responder", sub: "streams answer (Groq)", icon: "💬", kind: "agent" },
];

const EDGES: { a: AgentId; b: AgentId; sh: string; th: string }[] = [
  { a: "user", b: "supervisor", sh: "r", th: "l" },
  { a: "supervisor", b: "research", sh: "r", th: "l" },
  { a: "supervisor", b: "flight", sh: "r", th: "l" },
  { a: "supervisor", b: "hotel", sh: "r", th: "l" },
  { a: "supervisor", b: "booking", sh: "r", th: "l" },
  { a: "research", b: "vector", sh: "r", th: "l" },
  { a: "flight", b: "mcp", sh: "r", th: "l" },
  { a: "hotel", b: "mcp", sh: "r", th: "l" },
  { a: "booking", b: "mcp", sh: "r", th: "l" },
  { a: "supervisor", b: "responder", sh: "b", th: "t" },
  { a: "responder", b: "user", sh: "l", th: "b" },
];

export const edgeKey = (a: string, b: string) => [a, b].sort().join("~");

const STATUS_STYLE: Record<NodeStatus, string> = {
  idle: "border-slate-700 bg-slate-900/80 text-slate-300",
  active: "border-amber-400 bg-amber-500/15 text-amber-100 shadow-[0_0_24px_rgba(251,191,36,0.55)] animate-pulse",
  done: "border-emerald-500 bg-emerald-500/10 text-emerald-100",
};

function AgentNode({ data }: NodeProps<Node<AgentNodeData>>) {
  const kindRing = data.kind === "infra" ? "border-dashed" : "";
  return (
    <div className={`w-[170px] rounded-xl border-2 px-3 py-2 transition-all ${STATUS_STYLE[data.status]} ${kindRing}`}>
      {(["l", "r", "t", "b"] as const).map((h) => (
        <Handle
          key={h}
          id={h}
          type="source"
          position={{ l: Position.Left, r: Position.Right, t: Position.Top, b: Position.Bottom }[h]}
          className="!h-1.5 !w-1.5 !border-0 !bg-slate-500"
        />
      ))}
      <div className="flex items-center gap-2">
        <span className="text-xl">{data.icon}</span>
        <div className="min-w-0">
          <div className="truncate text-sm font-semibold">{data.label}</div>
          <div className="truncate text-[10px] opacity-70">{data.sub}</div>
        </div>
      </div>
    </div>
  );
}

const nodeTypes = { agent: AgentNode };

export default function AgentGraph({
  statuses,
  traversed,
  activeEdge,
}: {
  statuses: Partial<Record<AgentId, NodeStatus>>;
  traversed: Set<string>;
  activeEdge: { key: string; label: string } | null;
}) {
  const nodes: Node<AgentNodeData>[] = useMemo(
    () =>
      NODES.map((n) => ({
        id: n.id,
        type: "agent",
        position: { x: n.x, y: n.y },
        draggable: true,
        data: { label: n.label, sub: n.sub, icon: n.icon, kind: n.kind, status: statuses[n.id] ?? "idle" },
      })),
    [statuses],
  );

  const edges: Edge[] = useMemo(
    () =>
      EDGES.map((e) => {
        const key = edgeKey(e.a, e.b);
        const isActive = activeEdge?.key === key;
        const infraLive =
          (e.b === "mcp" || e.b === "vector") && statuses[e.b] === "active" && statuses[e.a] === "active";
        const live = isActive || infraLive;
        const seen = traversed.has(key) || (e.b === "mcp" || e.b === "vector" ? statuses[e.a] === "done" && statuses[e.b] === "done" : false);
        const color = live ? "#fbbf24" : seen ? "#10b981" : "#334155";
        return {
          id: key,
          source: e.a,
          target: e.b,
          sourceHandle: e.sh,
          targetHandle: e.th,
          animated: live,
          label: isActive ? truncate(activeEdge!.label, 40) : undefined,
          labelStyle: { fill: "#fde68a", fontSize: 10 },
          labelBgStyle: { fill: "#0f172a" },
          labelBgPadding: [4, 2] as [number, number],
          style: { stroke: color, strokeWidth: live ? 3 : 2 },
          markerEnd: { type: MarkerType.ArrowClosed, color },
        };
      }),
    [statuses, traversed, activeEdge],
  );

  return (
    <ReactFlow
      nodes={nodes}
      edges={edges}
      nodeTypes={nodeTypes}
      connectionMode={ConnectionMode.Loose}
      fitView
      fitViewOptions={{ padding: 0.12, minZoom: 0.2 }}
      minZoom={0.2}
      proOptions={{ hideAttribution: true }}
      nodesConnectable={false}
      colorMode="dark"
    >
      <Background color="#1e293b" gap={20} />
    </ReactFlow>
  );
}

function truncate(s: string, n: number) {
  return s.length > n ? s.slice(0, n - 1) + "…" : s;
}
