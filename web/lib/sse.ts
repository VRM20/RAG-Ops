// Incremental Server-Sent Events parser for fetch() streams.
//
// EventSource only supports GET, but /api/chat/stream is a POST, so we read the response
// body ourselves. Network chunks can split an event (or even a UTF-8 code point) at any
// byte, so we buffer decoded text and only emit events terminated by a blank line.

export interface SSEMessage {
  event: string;
  data: string;
  id?: string;
}

function parseBlock(block: string): SSEMessage | null {
  let event = "message";
  let id: string | undefined;
  const data: string[] = [];
  for (const rawLine of block.split("\n")) {
    if (!rawLine || rawLine.startsWith(":")) continue; // comment / keep-alive
    const colon = rawLine.indexOf(":");
    const field = colon === -1 ? rawLine : rawLine.slice(0, colon);
    let value = colon === -1 ? "" : rawLine.slice(colon + 1);
    if (value.startsWith(" ")) value = value.slice(1);
    if (field === "event") event = value;
    else if (field === "data") data.push(value);
    else if (field === "id") id = value;
  }
  if (data.length === 0 && event === "message") return null;
  return { event, data: data.join("\n"), id };
}

export async function* readSSE(response: Response, signal?: AbortSignal): AsyncGenerator<SSEMessage> {
  if (!response.body) throw new Error("Response has no body to stream");
  const reader = response.body.getReader();
  // `stream: true` keeps partial multi-byte sequences buffered between chunks.
  const decoder = new TextDecoder("utf-8");
  let buffer = "";
  try {
    while (true) {
      if (signal?.aborted) return;
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true }).replace(/\r\n?/g, "\n");
      let boundary = buffer.indexOf("\n\n");
      while (boundary !== -1) {
        const block = buffer.slice(0, boundary);
        buffer = buffer.slice(boundary + 2);
        const msg = parseBlock(block);
        if (msg) yield msg;
        boundary = buffer.indexOf("\n\n");
      }
    }
    buffer += decoder.decode();
    const tail = parseBlock(buffer.replace(/\r\n?/g, "\n").trim());
    if (tail) yield tail;
  } finally {
    reader.releaseLock();
  }
}
