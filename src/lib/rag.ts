// A tiny, dependency-free RAG pipeline: documents → chunks → embeddings → cosine search.
// Embeddings are TF-IDF vectors hashed into a fixed number of dimensions, so the demo
// runs with no external embedding API while showing exactly what a vector store does.

import type { RagChunk } from "./events";

const DIMS = 256;

const DOCS: { source: string; title: string; text: string }[] = [
  {
    source: "policies/cancellation.md",
    title: "Cancellation & Refund Policy",
    text: `Flights booked through SkyWay Travel can be cancelled free of charge within 24 hours of booking. After 24 hours, economy fares incur a cancellation fee of $150 per passenger; premium fares are fully refundable up to 7 days before departure.

Hotel reservations are free to cancel up to 48 hours before check-in. Late cancellations are charged one night's stay. Non-refundable "Saver" rates cannot be cancelled but can be changed once for a $25 fee.

Refunds are processed to the original payment method within 5-10 business days.`,
  },
  {
    source: "policies/baggage.md",
    title: "Baggage Policy",
    text: `Economy tickets include one cabin bag up to 7 kg and one personal item. Checked baggage of 23 kg is included on international routes and costs $35 on domestic routes.

Premium Economy and Business include two checked bags of 23 kg and 32 kg respectively. Sports equipment such as surfboards and golf bags must be pre-booked at least 48 hours before departure.`,
  },
  {
    source: "policies/payments.md",
    title: "Payments & Pricing",
    text: `We accept Visa, Mastercard, American Express, UPI and PayPal. Prices are shown in USD and include all taxes. A booking is confirmed once payment is authorised; you will receive a booking reference starting with SKY.

Price lock: you can hold a fare for 72 hours for a $10 fee, which is deducted from the final price when you book.`,
  },
  {
    source: "policies/insurance.md",
    title: "Travel Insurance",
    text: `SkyWay Protect travel insurance costs 6% of the trip value and covers trip cancellation for medical reasons, lost baggage up to $2,000, flight delays over 6 hours, and emergency medical expenses up to $100,000. Insurance must be purchased within 14 days of the initial booking.`,
  },
  {
    source: "policies/loyalty.md",
    title: "SkyWay Rewards Loyalty Program",
    text: `Members earn 5 points per dollar spent on flights and 10 points per dollar on hotels. Gold status is reached at 50,000 points and includes free seat selection, lounge access and priority check-in. 100 points can be redeemed for $1 off any future booking.`,
  },
  {
    source: "visa/japan.md",
    title: "Visa Guide: Japan",
    text: `Citizens of the US, UK, EU, Canada and Australia can visit Japan visa-free for up to 90 days for tourism. Indian citizens require a tourist visa, which can be applied for via the Japan eVISA system; processing takes about 5 working days. Passports must be valid for the duration of the stay.`,
  },
  {
    source: "visa/uae.md",
    title: "Visa Guide: United Arab Emirates",
    text: `US, UK and EU citizens receive a free 30-day visa on arrival in Dubai. Indian citizens holding a valid US visa or green card, or a UK/EU residence permit, can get a 14-day visa on arrival; others must apply for an e-visa online before travel, typically approved in 3-4 days.`,
  },
  {
    source: "visa/schengen.md",
    title: "Visa Guide: France & Schengen Area",
    text: `France is part of the Schengen Area. US, UK and Canadian citizens can stay up to 90 days in any 180-day period without a visa; from late 2026 travellers must register with ETIAS online before departure. Indian citizens need a Schengen short-stay visa, applied for at VFS Global at least 15 days before travel.`,
  },
  {
    source: "guides/paris.md",
    title: "Destination Guide: Paris",
    text: `Paris is best visited in spring (April-June) or early autumn (September-October). Must-sees include the Eiffel Tower, the Louvre, Montmartre and a Seine river cruise. The Paris Museum Pass gives access to over 50 museums. The Metro is the fastest way around the city; a Navigo weekly pass costs about €30.`,
  },
  {
    source: "guides/tokyo.md",
    title: "Destination Guide: Tokyo",
    text: `Tokyo shines in spring for cherry blossoms (late March-April) and in autumn (October-November) for colourful foliage. Top neighbourhoods are Shibuya, Shinjuku, Asakusa and Akihabara. A Suica card works on all trains and in convenience stores. Day trips to Mount Fuji, Nikko and Kamakura are popular.`,
  },
  {
    source: "guides/bali.md",
    title: "Destination Guide: Bali",
    text: `Bali's dry season runs from April to October and is the best time for beaches and hiking. Ubud is the cultural heart with rice terraces and temples, while Seminyak and Uluwatu are known for surfing and sunsets. Visitors pay a tourism levy of IDR 150,000 on arrival. Renting a scooter or hiring a driver is the usual way to get around.`,
  },
  {
    source: "guides/dubai.md",
    title: "Destination Guide: Dubai",
    text: `Dubai is most pleasant from November to March, when temperatures are around 25°C. Highlights include the Burj Khalifa, desert safaris, the Dubai Mall and the Gold Souk. Dress modestly in traditional areas. The Dubai Metro connects the airport with downtown and the Marina.`,
  },
  {
    source: "guides/new-york.md",
    title: "Destination Guide: New York",
    text: `New York City is great year-round, with fall (September-November) offering mild weather and the holiday season bringing festive lights. Don't miss Central Park, the Statue of Liberty, Broadway shows and the Metropolitan Museum. The subway runs 24/7; a 7-day unlimited MetroCard or OMNY tap is the easiest option.`,
  },
  {
    source: "guides/goa.md",
    title: "Destination Guide: Goa",
    text: `Goa's peak season is November to February with sunny, dry weather. North Goa (Baga, Anjuna) is lively with nightlife and markets; South Goa (Palolem, Colva) is quieter with pristine beaches. Old Goa has UNESCO-listed Portuguese churches. Monsoon (June-September) brings lower hotel prices and lush green landscapes.`,
  },
];

export type Chunk = { id: string; source: string; title: string; text: string; vector: number[] };

const STOP = new Set(
  "a an the and or of to in on for is are be by with at from as it this that can up per any into you your our we its".split(" "),
);

function tokenize(text: string): string[] {
  return text
    .toLowerCase()
    .replace(/[^a-z0-9\s]/g, " ")
    .split(/\s+/)
    .filter((t) => t.length > 1 && !STOP.has(t))
    .map((t) => t.replace(/(ing|ed|es|s)$/, ""));
}

function hash(token: string): number {
  let h = 2166136261;
  for (let i = 0; i < token.length; i++) {
    h ^= token.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  return Math.abs(h) % DIMS;
}

function chunkDocs(): Omit<Chunk, "vector">[] {
  const chunks: Omit<Chunk, "vector">[] = [];
  for (const doc of DOCS) {
    doc.text
      .split(/\n\s*\n/)
      .map((p) => p.trim())
      .filter(Boolean)
      .forEach((p, i) => chunks.push({ id: `${doc.source}#${i}`, source: doc.source, title: doc.title, text: p }));
  }
  return chunks;
}

type Store = { chunks: Chunk[]; idf: Map<string, number> };
let store: Store | null = null;

function embed(text: string, idf: Map<string, number>): number[] {
  const v = new Array(DIMS).fill(0);
  const tokens = tokenize(text);
  for (const t of tokens) v[hash(t)] += idf.get(t) ?? Math.log(2);
  const norm = Math.sqrt(v.reduce((s, x) => s + x * x, 0)) || 1;
  return v.map((x) => x / norm);
}

function getStore(): Store {
  if (store) return store;
  const raw = chunkDocs();
  const df = new Map<string, number>();
  for (const c of raw) for (const t of new Set(tokenize(c.title + " " + c.text))) df.set(t, (df.get(t) ?? 0) + 1);
  const idf = new Map<string, number>();
  for (const [t, n] of df) idf.set(t, Math.log(1 + raw.length / n));
  store = { idf, chunks: raw.map((c) => ({ ...c, vector: embed(c.title + " " + c.text, idf) })) };
  return store;
}

export function retrieve(query: string, k = 3) {
  const start = Date.now();
  const { chunks, idf } = getStore();
  const q = embed(query, idf);
  const scored: RagChunk[] = chunks
    .map((c) => ({
      id: c.id,
      source: c.source,
      title: c.title,
      text: c.text,
      score: c.vector.reduce((s, x, i) => s + x * q[i], 0),
    }))
    .sort((a, b) => b.score - a.score)
    .slice(0, k);
  // Show only non-zero dimensions so the "embedding" is readable in the UI.
  const preview = q
    .map((x, i) => [i, x] as const)
    .filter(([, x]) => x > 0)
    .slice(0, 12)
    .map(([, x]) => Number(x.toFixed(3)));
  return { chunks: scored, queryVector: preview, totalChunks: chunks.length, ms: Date.now() - start };
}

export const KNOWLEDGE_SOURCES = DOCS.map((d) => ({ source: d.source, title: d.title }));
