/** мм:сс (минуты не ограничиваются 59). */
export function formatClock(totalSec: number): string {
  const sec = Math.max(0, Math.floor(Number.isFinite(totalSec) ? totalSec : 0));
  const m = Math.floor(sec / 60);
  const s = sec % 60;
  return `${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}`;
}

/** Точные секунды: «44,2 с». */
export function formatSeconds(sec: number): string {
  return `${(Number.isFinite(sec) ? sec : 0).toLocaleString("ru-RU", { maximumFractionDigits: 2 })} с`;
}

export function formatSize(bytes: number): string {
  const units = ["Б", "КБ", "МБ", "ГБ"];
  let v = bytes;
  let i = 0;
  while (v >= 1024 && i < units.length - 1) {
    v /= 1024;
    i++;
  }
  return `${v.toLocaleString("ru-RU", { maximumFractionDigits: i === 0 ? 0 : 1 })} ${units[i]}`;
}

export function isMovFile(file: File): boolean {
  return /\.mov$/i.test(file.name) || file.type === "video/quicktime";
}

export function isHttpUrl(url: string | undefined | null): url is string {
  if (!url) return false;
  try {
    const u = new URL(url);
    return u.protocol === "http:" || u.protocol === "https:";
  } catch {
    return false;
  }
}
