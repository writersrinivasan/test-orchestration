// Events streamed from the server to the UI so every layer of the system is visible.

export type AgentId =
  | "user"
  | "supervisor"
  | "research"
  | "flight"
  | "hotel"
  | "booking"
  | "responder"
  | "mcp"
  | "vector";

export type RagChunk = {
  id: string;
  source: string;
  title: string;
  text: string;
  score: number;
};

export type TravelEvent =
  | { type: "node"; id: AgentId; status: "active" | "done"; detail?: string }
  | { type: "handoff"; from: AgentId; to: AgentId; message: string }
  | { type: "log"; agent: AgentId; text: string }
  | { type: "plan"; plan: string[]; intent: string; params: Record<string, unknown>; reasoning: string }
  | { type: "mcp_tools"; tools: { name: string; description?: string }[] }
  | { type: "mcp_wire"; direction: "client→server" | "server→client"; payload: unknown }
  | {
      type: "rag";
      query: string;
      queryVector: number[];
      chunks: RagChunk[];
      totalChunks: number;
      ms: number;
    }
  | { type: "token"; text: string }
  | { type: "final"; text: string; data: FinalData; runId?: string }
  | { type: "error"; message: string };

export type Flight = {
  id: string;
  airline: string;
  from: string;
  to: string;
  date: string;
  depart: string;
  arrive: string;
  stops: number;
  price: number;
};

export type Hotel = {
  id: string;
  name: string;
  city: string;
  stars: number;
  rating: number;
  pricePerNight: number;
  amenities: string[];
};

export type Booking = {
  reference: string;
  status: string;
  flight: Flight;
  hotel: Hotel;
  nights: number;
  guests: number;
  total: number;
};

export type FinalData = {
  flights?: Flight[];
  hotels?: Hotel[];
  booking?: Booking;
  weather?: Record<string, unknown>;
};

export type Emit = (e: TravelEvent) => void;
