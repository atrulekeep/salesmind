/**
 * BFF 反代逻辑：转发到后端 FastAPI（默认 http://127.0.0.1:8766）。
 * SSE（text/event-stream）以 ReadableStream 原样透传，避免跨域。
 */

const BACKEND = process.env.BACKEND_URL ?? "http://127.0.0.1:8766";

/** 只透传安全 header（fetch 已自动解压，content-encoding/length 不转发）。 */
function filterHeaders(upstream: Response): Headers {
  const headers = new Headers();
  for (const name of ["content-type", "cache-control", "x-accel-buffering"]) {
    const value = upstream.headers.get(name);
    if (value !== null) headers.set(name, value);
  }
  return headers;
}

export async function proxy(req: Request): Promise<Response> {
  const { pathname, search } = new URL(req.url);
  const hasBody = req.method !== "GET" && req.method !== "HEAD";
  const body = hasBody ? await req.arrayBuffer() : undefined;

  let upstream: Response;
  try {
    upstream = await fetch(`${BACKEND}${pathname}${search}`, {
      method: req.method,
      headers: { "content-type": req.headers.get("content-type") ?? "" },
      body,
      cache: "no-store",
    });
  } catch {
    return Response.json({ detail: "后端服务不可用" }, { status: 502 });
  }
  return new Response(upstream.body, {
    status: upstream.status,
    headers: filterHeaders(upstream),
  });
}
