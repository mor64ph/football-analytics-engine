import { useEffect, useState } from 'react';
import { api } from '../api/client';
import {
  Badge, Card, EmptyState, ErrorState, PageHeader,
  Segmented, Skeleton, Table, Td, Th,
} from '../components/ui';
import { competitionName, eur, num } from '../lib/format';

interface Mover {
  player_name: string;
  competition: string;
  current_value?: number;
  dynamic_value_eur?: number;
  predicted_value_eur?: number;
  prev_value?: number;
  delta_eur?: number;
  delta_pct?: number;
  form_score?: number;
  form_multiplier?: number;
  // Season-to-date returns: the free API tier exposes no per-match goal detail,
  // so form is a contribution rate over the season rather than a rolling window.
  goals_recent?: number;
  assists_recent?: number;
}

type Period = 'dod' | 'wow' | 'mom';
type Direction = 'top' | 'bottom';

/** Form as a proportion of the elite contribution rate the engine calibrates to. */
function FormBar({ score }: { score?: number }) {
  if (score === undefined || Number.isNaN(score)) {
    return <span className="text-ink-faint">—</span>;
  }
  const pct = Math.round(Math.max(0, Math.min(1, score)) * 100);
  const tone = pct > 60 ? 'bg-positive' : pct > 30 ? 'bg-caution' : 'bg-line-strong';
  return (
    <div className="flex items-center gap-2">
      <div className="h-1 w-16 rounded-full bg-surface-3">
        <div className={`h-full rounded-full ${tone}`} style={{ width: `${pct}%` }} />
      </div>
      <span className="tabular w-7 text-right text-micro text-ink-faint">{pct}%</span>
    </div>
  );
}

function DeltaCell({ pct }: { pct?: number }) {
  if (pct === undefined || Number.isNaN(pct)) {
    return <span className="text-ink-faint">—</span>;
  }
  const tone = pct > 0 ? 'text-positive' : pct < 0 ? 'text-negative' : 'text-ink-faint';
  return (
    <span className={`tabular font-medium ${tone}`}>
      {pct > 0 ? '+' : ''}{pct.toFixed(1)}%
    </span>
  );
}

export default function Leaderboard() {
  const [period, setPeriod] = useState<Period>('wow');
  const [direction, setDirection] = useState<Direction>('top');
  const [data, setData] = useState<Mover[]>([]);
  const [source, setSource] = useState('');
  const [note, setNote] = useState('');
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    setLoading(true);
    setError(null);
    api.getLeaderboard({ period, direction, limit: 50 })
      .then(json => {
        setData((json.data ?? []) as Mover[]);
        setSource(json.source ?? '');
        setNote(json.note ?? '');
      })
      .catch(e => setError(e?.message ?? 'Could not load movers.'))
      .finally(() => setLoading(false));
  }, [period, direction]);

  // Ranking by same-day form rather than by movement between two snapshots.
  // Worth saying plainly: the numbers mean something different.
  const isProxy = source === 'form_multiplier';

  return (
    <>
      <PageHeader
        title="Value Movers"
        subtitle="Transfer values adjusted for current scoring form and a contract-length discount, ranked by movement."
        action={
          <div className="flex items-center gap-2">
            <Segmented
              value={period}
              onChange={setPeriod}
              options={[
                { value: 'dod' as Period, label: 'Day' },
                { value: 'wow' as Period, label: 'Week' },
                { value: 'mom' as Period, label: 'Month' },
              ]}
            />
            <Segmented
              value={direction}
              onChange={setDirection}
              options={[
                { value: 'top' as Direction, label: 'Risers' },
                { value: 'bottom' as Direction, label: 'Fallers' },
              ]}
            />
          </div>
        }
      />

      {error && <ErrorState message={error} />}

      {isProxy && !loading && (
        <div className="rounded-md border border-caution/25 bg-caution/5 px-4 py-3 text-xs text-caution">
          Ranked by current form rather than movement between two days — only one
          snapshot exists so far. Day-to-day change appears once a second daily
          refresh has run.
        </div>
      )}

      <Card padded={false}>
        <div className="px-5 py-4">
          {loading ? (
            <Skeleton className="h-4 w-48" />
          ) : (
            <p className="text-xs text-ink-faint">
              {data.length ? `${num(data.length)} players` : 'No movement recorded'}
              {source && <> · source <span className="text-ink-muted">{source.replace('_', ' ')}</span></>}
            </p>
          )}
        </div>

        {loading ? (
          <div className="space-y-2 px-5 pb-5">
            {Array.from({ length: 8 }).map((_, i) => <Skeleton key={i} className="h-9 w-full" />)}
          </div>
        ) : data.length === 0 ? (
          <EmptyState
            title="Nothing to show yet"
            hint={note || 'Run the daily refresh to record a value snapshot.'}
          />
        ) : (
          <div className="px-5 pb-5">
            <Table>
              <thead>
                <tr>
                  <Th className="w-8">#</Th>
                  <Th>Player</Th>
                  <Th>League</Th>
                  <Th>Form</Th>
                  <Th align="right">G</Th>
                  <Th align="right">A</Th>
                  <Th align="right">Value</Th>
                  <Th align="right">Change</Th>
                </tr>
              </thead>
              <tbody>
                {data.map((row, i) => (
                  <tr key={`${row.player_name}-${i}`} className="transition-colors hover:bg-surface-2">
                    <Td className="text-ink-faint">{i + 1}</Td>
                    <Td className="font-medium text-ink">{row.player_name}</Td>
                    <Td>
                      <Badge>{competitionName(row.competition)}</Badge>
                    </Td>
                    <Td><FormBar score={row.form_score} /></Td>
                    <Td align="right" className="text-ink-muted">{row.goals_recent ?? 0}</Td>
                    <Td align="right" className="text-ink-muted">{row.assists_recent ?? 0}</Td>
                    <Td align="right" className="text-ink">
                      {eur(row.current_value ?? row.dynamic_value_eur)}
                    </Td>
                    <Td align="right"><DeltaCell pct={row.delta_pct} /></Td>
                  </tr>
                ))}
              </tbody>
            </Table>
          </div>
        )}
      </Card>
    </>
  );
}
