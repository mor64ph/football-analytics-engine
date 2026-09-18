import { useCallback, useEffect, useState } from 'react';
import { api } from '../api/client';
import {
  Badge, Card, EmptyState, PageHeader, Select, Skeleton, Table, Td, Th,
} from '../components/ui';
import { competitionName, eur, num, pct } from '../lib/format';
import type { Player } from '../types';

/** Error bands, ordered best to worst. Colour is semantic, not decorative. */
const BAND_TONE: Record<string, 'positive' | 'caution' | 'negative' | 'neutral'> = {
  '<10%': 'positive',
  '10-25%': 'positive',
  '25-50%': 'caution',
  '50-100%': 'caution',
  '>100%': 'negative',
};

const POSITIONS = [
  { value: '', label: 'All positions' },
  { value: 'GK', label: 'Goalkeeper' },
  { value: 'DEF', label: 'Defender' },
  { value: 'MID', label: 'Midfielder' },
  { value: 'ATT', label: 'Attacker' },
];

export default function PlayerSearch() {
  const [players, setPlayers] = useState<Player[]>([]);
  const [total, setTotal] = useState(0);
  const [search, setSearch] = useState('');
  const [comp, setComp] = useState('');
  const [pos, setPos] = useState('');
  const [loading, setLoading] = useState(true);
  const [competitions, setCompetitions] = useState<string[]>([]);

  useEffect(() => {
    api.getKPIs().then(k => setCompetitions(k.competitions)).catch(() => {});
  }, []);

  const load = useCallback(() => {
    setLoading(true);
    api.getPlayers({
      search: search || undefined,
      competition: comp || undefined,
      position: pos || undefined,
      limit: 100,
    })
      .then(r => { setPlayers(r.data); setTotal(r.total); })
      .catch(() => setPlayers([]))
      .finally(() => setLoading(false));
  }, [search, comp, pos]);

  // Debounced so typing a name does not fire a request per keystroke.
  useEffect(() => {
    const t = setTimeout(load, 250);
    return () => clearTimeout(t);
  }, [load]);

  return (
    <>
      <PageHeader
        title="Player Search"
        subtitle="Historical player-seasons: what the market paid against what the model would have paid."
        action={!loading
          ? <span className="text-xs text-ink-faint">{num(total)} matching records</span>
          : undefined}
      />

      <Card>
        <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-4">
          <div className="relative lg:col-span-2">
            <span className="pointer-events-none absolute left-3 top-1/2 -translate-y-1/2 text-ink-faint">
              <svg viewBox="0 0 16 16" className="size-4" fill="none" stroke="currentColor"
                   strokeWidth="1.5" strokeLinecap="round">
                <circle cx="7" cy="7" r="4.5" />
                <path d="M10.5 10.5 14 14" />
              </svg>
            </span>
            <input
              className="w-full rounded-md border border-line bg-surface-2 py-2 pl-9 pr-3 text-sm
                         text-ink placeholder:text-ink-faint transition-colors
                         hover:border-line-strong focus:border-brand focus:outline-none"
              placeholder="Search by player name…"
              value={search}
              onChange={e => setSearch(e.target.value)}
            />
          </div>

          <Select value={comp} onChange={setComp}>
            <option value="">All leagues</option>
            {competitions.map(c => (
              <option key={c} value={c}>{competitionName(c)}</option>
            ))}
          </Select>

          <Select value={pos} onChange={setPos}>
            {POSITIONS.map(p => <option key={p.value} value={p.value}>{p.label}</option>)}
          </Select>
        </div>
      </Card>

      <Card padded={false}>
        {loading ? (
          <div className="space-y-2 p-5">
            {Array.from({ length: 10 }).map((_, i) => <Skeleton key={i} className="h-9 w-full" />)}
          </div>
        ) : players.length === 0 ? (
          <EmptyState
            title="No players match those filters"
            hint="Try a shorter name fragment, or clear the league and position filters."
          />
        ) : (
          <div className="p-5">
            <Table>
              <thead>
                <tr>
                  <Th>Player</Th>
                  <Th>League</Th>
                  <Th align="right">Season</Th>
                  <Th align="right">Age</Th>
                  <Th align="right">G</Th>
                  <Th align="right">A</Th>
                  <Th align="right">Market</Th>
                  <Th align="right">Model</Th>
                  <Th align="right">Error</Th>
                </tr>
              </thead>
              <tbody>
                {players.map((p, i) => (
                  <tr key={`${p.player_name}-${p.season}-${i}`}
                      className="transition-colors hover:bg-surface-2">
                    <Td className="font-medium text-ink">{p.player_name}</Td>
                    <Td><Badge>{competitionName(p.competition)}</Badge></Td>
                    <Td align="right" className="text-ink-muted">{p.season}</Td>
                    <Td align="right" className="text-ink-muted">{p.age}</Td>
                    <Td align="right" className="text-ink-muted">{p.goals}</Td>
                    <Td align="right" className="text-ink-muted">{p.assists}</Td>
                    <Td align="right" className="text-ink">{eur(p.market_value_eur)}</Td>
                    <Td align="right" className="text-ink">{eur(p.predicted_value_eur)}</Td>
                    <Td align="right">
                      <Badge tone={BAND_TONE[p.accuracy_band] ?? 'neutral'}>
                        {pct(p.abs_pct_error, 0)}
                      </Badge>
                    </Td>
                  </tr>
                ))}
              </tbody>
            </Table>
            {total > players.length && (
              <p className="mt-4 border-t border-line pt-4 text-xs text-ink-faint">
                Showing the first {num(players.length)} of {num(total)}. Narrow the search to see more.
              </p>
            )}
          </div>
        )}
      </Card>
    </>
  );
}
