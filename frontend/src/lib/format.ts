/**
 * Number formatting, defined once.
 *
 * Currency and percentages were previously formatted inline at each call site,
 * so the same value appeared as "€44.2M", "€44M" and "44.2" on different
 * screens. Money in particular needs consistent rules about when to switch
 * units, or columns stop lining up.
 */

/** Compact euro. Switches units so a column never mixes magnitudes awkwardly. */
export function eur(value: number | null | undefined, opts?: { sign?: boolean }): string {
  if (value === null || value === undefined || Number.isNaN(value)) return '—';
  const sign = opts?.sign && value > 0 ? '+' : value < 0 ? '−' : '';
  const abs = Math.abs(value);

  if (abs >= 1e9) return `${sign}€${(abs / 1e9).toFixed(2)}bn`;
  if (abs >= 1e8) return `${sign}€${Math.round(abs / 1e6)}M`;
  if (abs >= 1e6) return `${sign}€${(abs / 1e6).toFixed(1)}M`;
  if (abs >= 1e3) return `${sign}€${Math.round(abs / 1e3)}k`;
  return `${sign}€${Math.round(abs)}`;
}

/** Euro in millions, for axis ticks where the unit is stated once on the axis. */
export function eurM(value: number): string {
  return `€${(value / 1e6).toFixed(0)}M`;
}

export function pct(value: number | null | undefined, digits = 1): string {
  if (value === null || value === undefined || Number.isNaN(value)) return '—';
  return `${value.toFixed(digits)}%`;
}

/** A 0–1 probability as a percentage. */
export function prob(value: number | null | undefined, digits = 1): string {
  if (value === null || value === undefined || Number.isNaN(value)) return '—';
  return `${(value * 100).toFixed(digits)}%`;
}

export function num(value: number | null | undefined): string {
  if (value === null || value === undefined || Number.isNaN(value)) return '—';
  return value.toLocaleString();
}

export const COMPETITIONS: Record<string, { name: string; short: string }> = {
  PL:  { name: 'Premier League', short: 'PL' },
  PD:  { name: 'La Liga',        short: 'LaLiga' },
  SA:  { name: 'Serie A',        short: 'Serie A' },
  BL1: { name: 'Bundesliga',     short: 'Bundesliga' },
  FL1: { name: 'Ligue 1',        short: 'Ligue 1' },
};

export const competitionName = (code: string) =>
  COMPETITIONS[code]?.name ?? code;
