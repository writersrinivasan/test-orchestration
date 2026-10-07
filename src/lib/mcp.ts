// A real Model Context Protocol server ("travel-tools") and client, wired together with the
// SDK's in-memory transport. Every JSON-RPC message on the wire is emitted to the UI.

import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { InMemoryTransport } from "@modelcontextprotocol/sdk/inMemory.js";
import { z } from "zod";
import { traceable } from "langsmith/traceable";
import type { Booking, Emit, Flight, Hotel } from "./events";

const AIRLINES = ["SkyWay Air", "IndiGo", "Emirates", "Air France", "ANA", "Singapore Airlines", "Delta"];
const HOTEL_PREFIX = ["Grand", "Royal", "Azure", "Urban", "Lotus", "Harbor", "Palm"];
const HOTEL_SUFFIX = ["Palace", "Suites", "Resort & Spa", "Boutique Hotel", "Residency", "Inn"];
const AMENITIES = ["Free WiFi", "Pool", "Breakfast", "Spa", "Gym", "Airport shuttle", "Sea view", "Rooftop bar"];

// Deterministic pseudo-random numbers so the same search gives the same results.
function seeded(seed: string) {
  let h = 1779033703 ^ seed.length;
  for (let i = 0; i < seed.length; i++) h = Math.imul(h ^ seed.charCodeAt(i), 3432918353);
  return () => {
    h = Math.imul(h ^ (h >>> 16), 2246822507);
    h = Math.imul(h ^ (h >>> 13), 3266489909);
    return ((h ^= h >>> 16) >>> 0) / 4294967296;
  };
}

const cap = (s: string) => s.replace(/\b\w/g, (c) => c.toUpperCase());
const pad = (n: number) => String(n).padStart(2, "0");

function searchFlights(from: string, to: string, date: string): Flight[] {
  const r = seeded(`${from}-${to}-${date}`.toLowerCase());
  return Array.from({ length: 5 }, (_, i) => {
    const dep = 5 + Math.floor(r() * 17);
    const dur = 2 + Math.floor(r() * 12);
    const stops = r() > 0.6 ? 1 : 0;
    return {
      id: `FL${100 + i}${Math.floor(r() * 900)}`,
      airline: AIRLINES[Math.floor(r() * AIRLINES.length)],
      from: cap(from),
      to: cap(to),
      date,
      depart: `${pad(dep)}:${pad(Math.floor(r() * 4) * 15)}`,
      arrive: `${pad((dep + dur + stops * 2) % 24)}:${pad(Math.floor(r() * 4) * 15)}`,
      stops,
      price: Math.round(180 + r() * 900 - stops * 60),
    };
  }).sort((a, b) => a.price - b.price);
}

function searchHotels(city: string, maxPrice?: number): Hotel[] {
  const r = seeded(city.toLowerCase());
  const hotels = Array.from({ length: 6 }, (_, i) => {
    const stars = 3 + Math.floor(r() * 3);
    return {
      id: `HT${200 + i}`,
      name: `${HOTEL_PREFIX[Math.floor(r() * HOTEL_PREFIX.length)]} ${cap(city)} ${HOTEL_SUFFIX[Math.floor(r() * HOTEL_SUFFIX.length)]}`,
      city: cap(city),
      stars,
      rating: Number((3.8 + r() * 1.2).toFixed(1)),
      pricePerNight: Math.round(40 + stars * 35 + r() * 120),
      amenities: AMENITIES.filter(() => r() > 0.5).slice(0, 4),
    };
  });
  return hotels.filter((h) => !maxPrice || h.pricePerNight <= maxPrice).sort((a, b) => b.rating - a.rating);
}

function getWeather(city: string, date: string) {
  const r = seeded(`${city}${date}`);
  const conditions = ["Sunny", "Partly cloudy", "Light showers", "Clear skies", "Warm & humid"];
  return {
    city: cap(city),
    date,
    condition: conditions[Math.floor(r() * conditions.length)],
    highC: Math.round(18 + r() * 15),
    lowC: Math.round(10 + r() * 10),
  };
}

const bookings = new Map<string, Booking>();

function createServer() {
  const server = new McpServer({ name: "travel-tools", version: "1.0.0" });
  const json = (data: unknown) => ({ content: [{ type: "text" as const, text: JSON.stringify(data) }] });

  server.registerTool(
    "search_flights",
    {
      description: "Search available flights between two cities on a date",
      inputSchema: { from: z.string(), to: z.string(), date: z.string().describe("YYYY-MM-DD") },
    },
    async ({ from, to, date }) => json(searchFlights(from, to, date)),
  );

  server.registerTool(
    "search_hotels",
    {
      description: "Search hotels in a city, optionally under a max nightly price (USD)",
      inputSchema: { city: z.string(), maxPrice: z.number().optional() },
    },
    async ({ city, maxPrice }) => json(searchHotels(city, maxPrice)),
  );

  server.registerTool(
    "get_weather",
    {
      description: "Get the weather forecast for a city on a date",
      inputSchema: { city: z.string(), date: z.string() },
    },
    async ({ city, date }) => json(getWeather(city, date)),
  );

  server.registerTool(
    "create_booking",
    {
      description: "Create a confirmed booking for a flight + hotel package",
      inputSchema: {
        flight: z.any(),
        hotel: z.any(),
        nights: z.number(),
        guests: z.number(),
      },
    },
    async ({ flight, hotel, nights, guests }) => {
      const f = flight as Flight;
      const h = hotel as Hotel;
      const booking: Booking = {
        reference: `SKY${Math.random().toString(36).slice(2, 8).toUpperCase()}`,
        status: "CONFIRMED",
        flight: f,
        hotel: h,
        nights,
        guests,
        total: f.price * guests + h.pricePerNight * nights,
      };
      bookings.set(booking.reference, booking);
      return json(booking);
    },
  );

  return server;
}

/** Opens an MCP session and mirrors every JSON-RPC message to the UI. */
export async function openMcpSession(emit: Emit) {
  const [clientTransport, serverTransport] = InMemoryTransport.createLinkedPair();

  const clientSend = clientTransport.send.bind(clientTransport);
  clientTransport.send = async (msg, opts) => {
    emit({ type: "mcp_wire", direction: "client→server", payload: msg });
    return clientSend(msg, opts);
  };
  const serverSend = serverTransport.send.bind(serverTransport);
  serverTransport.send = async (msg, opts) => {
    emit({ type: "mcp_wire", direction: "server→client", payload: msg });
    return serverSend(msg, opts);
  };

  const server = createServer();
  const client = new Client({ name: "travel-agents", version: "1.0.0" });
  await server.connect(serverTransport);
  await client.connect(clientTransport);

  const { tools } = await client.listTools();
  emit({ type: "mcp_tools", tools: tools.map((t) => ({ name: t.name, description: t.description })) });

  return {
    async call<T>(name: string, args: Record<string, unknown>): Promise<T> {
      // Shows up in LangSmith as a tool span under the calling agent (no-op when tracing is off).
      const traced = traceable(
        async (toolArgs: Record<string, unknown>) => {
          const res = await client.callTool({ name, arguments: toolArgs });
          const content = res.content as { type: string; text: string }[];
          if (res.isError) throw new Error(`MCP tool ${name} failed: ${content[0]?.text}`);
          return JSON.parse(content[0].text) as T;
        },
        { name: `mcp.${name}`, run_type: "tool", metadata: { mcpServer: "travel-tools" } },
      );
      return traced(args);
    },
    async close() {
      await client.close();
      await server.close();
    },
  };
}

export type McpSession = Awaited<ReturnType<typeof openMcpSession>>;
