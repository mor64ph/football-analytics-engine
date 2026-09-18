import { useEffect, useState } from 'react';
import {
  Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis,
} from 'recharts';
import { api } from '../api/client';
import {
  Card, CardHeader, PageHeader, Skeleton, Table, Td, Th,
} from '../components/ui';
import { CHART, ChartTooltip, axisProps, gridProps } from '../lib/chart';
import { competitionName, eur, num, pct } from '../lib/format';
import type { LeagueStat } from '../types';

export default function LeagueComparison() {
  const [stats, setStats] = useState<LeagueStat[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    api.getLeagues()
      .then(r => setStats(r.data))
      .catch(() => setStats([]))
      .finally(() => setLoading(false));
  }, []);

  const valueData = stats.map(s => ({
    name: competitionName(s.competition),
    actual: s.avg_actual_value,
    predicted: s.avg_predicted_value,
  }));

  const fitData = stats.map(s => ({
    name: competitionName(s.competition),
    mape: s.avg_mape,
    within25: s.within_25_pct,
  }));

  return (
    <>
      <PageHeader
        title="League Comparison"
        subtitle="How valuations and model fit differ across Europe's top five divisions."
      />

      <div className="grid grid-cols-1 gap-6 xl:grid-cols-2">
        <Card>
          <CardHeader
            title="Average value: market against model"
            hint="Mean player valuation per league. A consistent gap in one direction means the model carries a league-level bias."
          />
          {loading ? <Skeleton className="h-[260px] w-full" /> : (
            <>
              <ResponsiveContainer width="100%" height={260}>
                <BarChart data={valueData} margin={{ top: 8, right: 8, left: 4, bottom: 0 }}>
                  <CartesianGrid {...gridProps} />
                  <XAxis dataKey="name" {...axisProps} />
                  <YAxis tickFormatter={v => `€${(v / 1e6).toFixed(0)}M`} width={56} {...axisProps} />
                  <Tooltip
                    cursor={{ fill: 'rgb(255 255 255 / 0.03)' }}
                    content={({ active, payload, label }) =>
                      active && payload?.length ? (
                        <ChartTooltip
                          title={label}
                          rows={[
                            { label: 'Market', value: eur(payload[0].payload.actual), color: CHART.series[1] },
                            { label: 'Model', value: eur(payload[0].payload.predicted), color: CHART.series[0] },
                          ]}
                        />
                      ) : null
                    }
                  />
                  <Bar dataKey="actual" fill={CHART.series[1]} radius={[3, 3, 0, 0]} barSize={18} />
                  <Bar dataKey="predicted" fill={CHART.series[0]} radius={[3, 3, 0, 0]} barSize={18} />
                </BarChart>
              </ResponsiveContainer>
              <div className="mt-2 flex justify-center gap-4 text-micro text-ink-faint">
                <span className="flex items-center gap-1.5">
                  <span className="size-2 rounded-full" style={{ background: CHART.series[1] }} />Market
                </span>
                <span className="flex items-center gap-1.5">
                  <span className="size-2 rounded-full" style={{ background: CHART.series[0] }} />Model
                </span>
              </div>
            </>
          )}
        </Card>

        <Card>
          <CardHeader
            title="Where the model fits best"
            hint="Share of predictions landing within 25% of the market price. Higher is better; the leagues with the most valuation churn are hardest."
          />
          {loading ? <Skeleton className="h-[260px] w-full" /> : (
            <ResponsiveContainer width="100%" height={260}>
              <BarChart data={fitData} layout="vertical"
                        margin={{ top: 0, right: 16, left: 8, bottom: 0 }}>
                <CartesianGrid {...gridProps} vertical horizontal={false} />
                <XAxis type="number" unit="%" {...axisProps} />
                <YAxis type="category" dataKey="name" width={92}
                       tick={{ fill: '#9ba3af', fontSize: 11 }}
                       tickLine={false} axisLine={false} />
                <Tooltip
                  cursor={{ fill: 'rgb(255 255 255 / 0.03)' }}
                  content={({ active, payload, label }) =>
                    active && payload?.length ? (
                      <ChartTooltip
                        title={label}
                        rows={[
                          { label: 'Within 25%', value: pct(payload[0].payload.within25), color: CHART.positive },
                          { label: 'Mean error', value: pct(payload[0].payload.mape) },
                        ]}
                      />
                    ) : null
                  }
                />
                <Bar dataKey="within25" fill={CHART.positive} radius={[0, 3, 3, 0]} barSize={16} />
              </BarChart>
            </ResponsiveContainer>
          )}
        </Card>
      </div>

      <Card>
        <CardHeader title="By the numbers" />
        {loading ? (
          <div className="space-y-2">
            {Array.from({ length: 5 }).map((_, i) => <Skeleton key={i} className="h-9 w-full" />)}
          </div>
        ) : (
          <Table>
            <thead>
              <tr>
                <Th>League</Th>
                <Th align="right">Players</Th>
                <Th align="right">Avg market</Th>
                <Th align="right">Avg model</Th>
                <Th align="right">Mean error</Th>
                <Th align="right">Within 25%</Th>
              </tr>
            </thead>
            <tbody>
              {stats.map(s => (
                <tr key={s.competition} className="transition-colors hover:bg-surface-2">
                  <Td className="font-medium text-ink">{competitionName(s.competition)}</Td>
                  <Td align="right" className="text-ink-muted">{num(s.total_players)}</Td>
                  <Td align="right" className="text-ink">{eur(s.avg_actual_value)}</Td>
                  <Td align="right" className="text-ink">{eur(s.avg_predicted_value)}</Td>
                  <Td align="right" className="text-ink-muted">{pct(s.avg_mape)}</Td>
                  <Td align="right" className="font-medium text-positive">{pct(s.within_25_pct)}</Td>
                </tr>
              ))}
            </tbody>
          </Table>
        )}
      </Card>
    </>
  );
}
