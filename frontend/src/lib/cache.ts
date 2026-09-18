const TTL_MS = 60 * 60 * 1000; // 1 hour — data refreshes daily, so this is safe

interface CacheEntry<T> {
  value: T;
  expires: number;
}

export function getLocal<T>(key: string): T | null {
  try {
    const raw = localStorage.getItem(`scoutiq:${key}`);
    if (!raw) return null;
    const entry: CacheEntry<T> = JSON.parse(raw);
    return Date.now() < entry.expires ? entry.value : null;
  } catch {
    return null;
  }
}

export function setLocal<T>(key: string, value: T): void {
  try {
    localStorage.setItem(`scoutiq:${key}`, JSON.stringify({
      value,
      expires: Date.now() + TTL_MS,
    }));
  } catch {
    // Quota exceeded or private-mode restriction — silently ignore.
  }
}
