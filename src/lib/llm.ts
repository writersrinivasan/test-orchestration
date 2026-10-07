import { ChatGroq } from "@langchain/groq";
import { HumanMessage, SystemMessage } from "@langchain/core/messages";

export const hasGroqKey = () => Boolean(process.env.GROQ_API_KEY);

export function getModel(temperature = 0) {
  return new ChatGroq({
    apiKey: process.env.GROQ_API_KEY,
    model: process.env.GROQ_MODEL || "openai/gpt-oss-120b",
    temperature,
    maxRetries: 4, // rides out Groq free-tier tokens-per-minute limits instead of failing the step
  });
}

/** Ask the model for a JSON object. Returns null without a key or on a parse failure. */
export async function llmJSON<T>(system: string, user: string): Promise<T | null> {
  if (!hasGroqKey()) return null;
  try {
    const res = await getModel().invoke([
      new SystemMessage(system + "\nRespond with a single JSON object only. No prose, no code fences."),
      new HumanMessage(user),
    ]);
    const text = String(res.content);
    const match = text.match(/\{[\s\S]*\}/);
    return match ? (JSON.parse(match[0]) as T) : null;
  } catch (err) {
    console.error("llmJSON failed", err);
    return null;
  }
}
