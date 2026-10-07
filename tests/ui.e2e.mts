// Browser end-to-end test using the locally installed Google Chrome.
// Run: BASE=http://localhost:3100 npm run test:ui
import { chromium } from "playwright-core";
import { writeFileSync } from "node:fs";

const BASE = process.env.BASE ?? "http://localhost:3000";
const OUT = new URL("./results/", import.meta.url).pathname;
const CHROME = process.env.CHROME ?? "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome";

const results: { check: string; pass: boolean; detail?: string }[] = [];
const check = (name: string, pass: boolean, detail?: string) => {
  results.push({ check: name, pass, detail });
  console.log(`${pass ? "PASS" : "FAIL"} ${name}${detail ? ` — ${detail}` : ""}`);
};

const browser = await chromium.launch({ executablePath: CHROME, headless: true });
const metrics: Record<string, unknown> = {};
try {
  const page = await browser.newPage({ viewport: { width: 1500, height: 900 } });
  const consoleErrors: string[] = [];
  page.on("console", (m) => m.type() === "error" && consoleErrors.push(m.text()));
  page.on("pageerror", (e) => consoleErrors.push(e.message));

  // ---- Load ----
  await page.goto(BASE, { waitUntil: "networkidle" });
  const nav = await page.evaluate(() => {
    const n = performance.getEntriesByType("navigation")[0] as PerformanceNavigationTiming;
    const fcp = performance.getEntriesByName("first-contentful-paint")[0];
    return { ttfb: n.responseStart, domContentLoaded: n.domContentLoadedEventEnd, load: n.loadEventEnd, fcp: fcp?.startTime };
  });
  metrics.pageLoad = Object.fromEntries(Object.entries(nav).map(([k, v]) => [k, Math.round(Number(v))]));
  check("Page title", (await page.title()) === "SkyWay Agentic Travel");
  check("4 suggestion prompts shown", (await page.locator("button", { hasText: /flights|policy|visa|Book/ }).count()) === 4);
  await page.waitForSelector(".react-flow__node");
  check("9 workflow nodes rendered", (await page.locator(".react-flow__node").count()) === 9);
  check("11 workflow edges rendered", (await page.locator(".react-flow__edge").count()) === 11);
  check("MCP / RAG / Agent panels present", (await page.getByText("MCP Layer").count()) > 0 && (await page.getByText("RAG Layer").count()) > 0 && (await page.getByText("Agent Messages").count()) > 0);
  await page.screenshot({ path: OUT + "ui-1-idle.png" });

  // ---- Full run: RAG + MCP + booking ----
  const q = "Book a trip from Mumbai to Paris for 2 guests, 5 nights and tell me the cancellation policy";
  await page.fill("input", q);
  const t0 = Date.now();
  await page.click("button:has-text('Send')");
  await page.waitForSelector(".animate-pulse.border-amber-400, .border-amber-400", { timeout: 15000 });
  const sawActive = Date.now() - t0;
  await page.waitForTimeout(2500);
  await page.screenshot({ path: OUT + "ui-2-running.png" });
  check("Live 'working' highlight appears during run", true, `first highlight after ${sawActive} ms`);
  await page.waitForSelector("text=Booking CONFIRMED", { timeout: 90000 });
  await page.waitForSelector("button:has-text('Send'):not([disabled])", { timeout: 30000 });
  metrics.uiRunMs = Date.now() - t0;
  await page.waitForTimeout(800);
  await page.screenshot({ path: OUT + "ui-3-complete.png" });

  const doneNodes = await page.locator(".react-flow__node .border-emerald-500").count();
  check("All visited nodes turn green (done)", doneNodes >= 8, `${doneNodes}/9 green`);
  const ref = await page.locator("text=/SKY[A-Z0-9]{6}/").first().textContent();
  check("Booking card shows SKY reference", /SKY[A-Z0-9]{6}/.test(ref ?? ""), ref ?? "");
  check("Flight cards shown", (await page.getByText("✈️ Flights").count()) > 0);
  check("Hotel cards shown", (await page.getByText("🏨 Hotels").count()) > 0);
  const toolChips = await page.locator("span.font-mono", { hasText: /^(search_flights|search_hotels|get_weather|create_booking)$/ }).count();
  check("MCP panel lists 4 discovered tools", toolChips === 4, `${toolChips} chips`);
  const wire = await page.locator("details").count();
  check("MCP panel shows JSON-RPC wire messages", wire >= 10, `${wire} messages`);
  const ragHits = await page.locator("text=/^#\\d /").count();
  check("RAG panel shows retrieved chunks with scores", ragHits === 3, `${ragHits} chunks`);
  const logLines = await page.locator("text=/^\\+\\d+\\.\\d+s$/").count();
  check("Agent message timeline populated", logLines >= 15, `${logLines} entries`);
  check("Supervisor plan overlay shown", (await page.getByText("Supervisor plan").count()) > 0);
  await page.locator("details").filter({ hasText: "tools/call" }).first().click();
  check("Wire message expands to JSON payload", (await page.locator("details[open] pre").first().textContent())?.includes('"jsonrpc": "2.0"') ?? false);

  // ---- XSS: user and model text must render as text ----
  await page.fill("input", `<img src=x onerror="window.__xss=1"> hotels in Goa`);
  await page.click("button:has-text('Send')");
  await page.waitForSelector("button:has-text('Send'):not([disabled])", { timeout: 90000 });
  check("XSS payload rendered inert", (await page.evaluate(() => (window as unknown as { __xss?: number }).__xss)) === undefined);

  check("No browser console errors", consoleErrors.length === 0, consoleErrors.slice(0, 3).join(" | "));

  // ---- Responsive ----
  const mobile = await browser.newPage({ viewport: { width: 390, height: 844 } });
  await mobile.goto(BASE, { waitUntil: "networkidle" });
  // Every panel must sit inside the 390px width (stacked vertically), not be clipped off-screen.
  const boxes = await mobile.evaluate(() =>
    [".react-flow", "input"].concat(["MCP Layer", "RAG Layer", "Agent Messages"]).map((sel) => {
      const el = sel.startsWith(".") || sel === "input"
        ? document.querySelector(sel)
        : [...document.querySelectorAll("div")].find((d) => d.children.length === 0 && d.textContent?.endsWith(sel));
      const r = el?.getBoundingClientRect();
      return { sel, left: r?.left ?? -1, right: r?.right ?? 9999, width: r?.width ?? 0 };
    }),
  );
  const clipped = boxes.filter((b) => b.left < 0 || b.right > 391 || b.width === 0);
  await mobile.screenshot({ path: OUT + "ui-4-mobile.png", fullPage: true });
  check("Mobile (390px): chat, graph, MCP, RAG, timeline all within viewport", clipped.length === 0, clipped.map((c) => c.sel).join(", "));
} finally {
  await browser.close();
}
writeFileSync(OUT + "ui.json", JSON.stringify({ results, metrics }, null, 2));
console.log(metrics);
