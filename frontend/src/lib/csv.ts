export function toCSV(rows: any[], columns: string[]): string {
  const esc = (v: any) => {
    const s = (v ?? "").toString();
    return /[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
  };
  const head = columns.join(",");
  const body = rows.map(r => columns.map(c => esc(r[c])).join(",")).join("\n");
  return head + "\n" + body;
}
