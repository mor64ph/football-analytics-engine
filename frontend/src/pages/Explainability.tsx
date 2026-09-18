import { useEffect, useState } from 'react';
import {
  Bar, BarChart, CartesianGrid, Cell, ResponsiveContainer,
  Scatter, ScatterChart, Tooltip, XAxis, YAxis, ZAxis,
} from 'recharts';
import { api } from '../api/client';
import {
  Card, CardHeader, PageHeader, Skeleton, Stat,
} from '../components/ui';
import { CHART, ChartTooltip, axisProps, gridProps } from '../lib/chart';
import { eur, num, pct } from '../lib/format';
import type { AccuracyBand, FeatureImportance, KPIs, Player } from '../types';

const FEAT_LABELS: Record<string, string> = {
  club_prestige_eur:  'Club prestige',
  international_caps: 'International caps',
  age_factor:         'Age factor',
  age:                'Age',
  league_difficulty:  'League strength',
  minutes_played:     'Minutes played',
  gc_p90:             'Goal contributions / 90',
  assists_p90:        'Assists / 90',
  goals_p90:          'Goals / 90',
};

const POSITION_COLORS: Record<string, string> = {
  ATT: CHART.series[0], MID: CHART.series[1], DEF: CHART.series[2], GK: CHART.series[3],
};

export default function Explainability() {
  const [importance, setImportance] = useState<FeatureImportance[]>([]);
  const [players, setPlayers] = useState<Player[]>([]);
  const [bands, setBands] = useState<AccuracyBand[]>([]);
  const [kpis, setKpis] = useState<KPIs | null>(null);

  useEffect(() => {
    api.getImportance().then(r => setImportance(r.data)).catch(() => {});
    api.getPlayers({ limit: 1200 }).then(r => setPlayers(r.data)).catch(() => {});
    api.getAccuracyBands().then(r => setBands(r.data)).catch(() => {});
    api.getKPIs().then(setKpis).catch(() => {});
  }, []);

  const impData = importance
    .filter(f => f.importance > 0.001)
    .map(f => ({ ...f, label: FEAT_LABELS[f.feature] ?? f.feature }))
    .sort((a, b) => a.importance - b.importance);

  const fit = players.slice(0, 900).map(p => ({
    actual: p.market_value_eur / 1e6,
    predicted: p.predicted_value_eur / 1e6,
    position: p.position_group,
    name: p.player_name,
  }));

  const bandTotal = bands.reduce((s, b) => s + b.count, 0) || 1;
  const bandTone = (band: string) =>
    band === '<10%' || band === '10-25%' ? CHART.positive
      : band === '>100%' ? CHART.negative : CHART.caution;

  return (
    <>
      <PageHeader
        title="Model Insights"
        subtitle="What drives a predicted valuation, how closely it tracks the market, and where it does not."
      />

      <div className="grid grid-cols-1 gap-4 md:grid-cols-3">
        <Stat label="Mean absolute error"
              value={kpis ? pct(kpis.avg_mape) : null}
              detail="Averaged across every player-season" tone="caution" />
        <Stat label="Within 25%"
              value={kpis ? pct(kpis.within_25_pct) : null}
              detail="Predictions close to the market price" tone="positive" />
        <Stat label="Training records"
              value={kpis ? num(kpis.total_records) : null}
              detail={kpis ? `${kpis.seasons_covered[0]}–${kpis.seasons_covered.at(-1)}, ${kpis.competitions.length} leagues` : undefined} />
      </div>

      <Card>
        <CardHeader
          title="What the model actually uses"
          hint="Gain-based importance, averaged over the four position models. Reputation and context outweigh a player's own output — which is a finding about the transfer market, not a flaw in the fit."
        />
        {impData.length ? (
          <ResponsiveContainer width="100%" height={Math.max(impData.length * 30, 240)}>
            <BarChart data={impData} layout="vertical"
                      margin={{ top: 0, right: 40, left: 8, bottom: 0 }}>
              <CartesianGrid {...gridProps} vertical horizontal={false} />
              <XAxis type="number" tickFormatter={v => `${(v * 100).toFixed(0)}%`} {...axisProps} />
              <YAxis type="category" dataKey="label" width={150}
                     tick={{ fill: '#9ba3af', fontSize: 11 }}
                     tickLine={false} axisLine={false} />
              <Tooltip
                cursor={{ fill: 'rgb(255 255 255 / 0.03)' }}
                content={({ active, payload, label }) =>
                  active && payload?.length ? (
                    <ChartTooltip
                      title={label}
                      rows={[{ label: 'Share of prediction',
                               value: pct(Number(payload[0].value) * 100),
                               color: CHART.series[0] }]}
                    />
                  ) : null
                }
              />
              <Bar dataKey="importance" radius={[0, 3, 3, 0]} barSize={14}
                   fill={CHART.series[0]} />
            </BarChart>
          </ResponsiveContainer>
        ) : <Skeleton className="h-[300px] w-full" />}
      </Card>

      <div className="grid grid-cols-1 gap-6 xl:grid-cols-3">
        <Card className="xl:col-span-2">
          <CardHeader
            title="Predicted against market value"
            hint="Each point is a player-season. On the diagonal the model agreed with the market; above it the model paid more, below it less."
          />
          {fit.length ? (
            <>
              <ResponsiveContainer width="100%" height={320}>
                <ScatterChart margin={{ top: 8, right: 12, left: 4, bottom: 4 }}>
                  <CartesianGrid {...gridProps} vertical />
                  <XAxis type="number" dataKey="actual" name="Market" unit="M" {...axisProps} />
                  <YAxis type="number" dataKey="predicted" name="Model" unit="M" {...axisProps} />
                  <ZAxis range={[16, 16]} />
                  <Tooltip
                    cursor={{ strokeDasharray: '3 3', stroke: CHART.grid }}
                    content={({ active, payload }) =>
                      active && payload?.length ? (
                        <ChartTooltip
                          title={payload[0].payload.name}
                          rows={[
                            { label: 'Market', value: eur(payload[0].payload.actual * 1e6) },
                            { label: 'Model', value: eur(payload[0].payload.predicted * 1e6),
                              color: POSITION_COLORS[payload[0].payload.position] },
                          ]}
                        />
                      ) : null
                    }
                  />
                  <Scatter data={fit} fillOpacity={0.55}>
                    {fit.map((d, i) => (
                      <Cell key={i} fill={POSITION_COLORS[d.position] ?? CHART.axis} />
                    ))}
                  </Scatter>
                </ScatterChart>
              </ResponsiveContainer>
              <div className="mt-3 flex flex-wrap justify-center gap-4 text-micro text-ink-faint">
                {Object.entries(POSITION_COLORS).map(([pos, c]) => (
                  <span key={pos} className="flex items-center gap-1.5">
                    <span className="size-2 rounded-full" style={{ background: c }} />{pos}
                  </span>
                ))}
              </div>
            </>
          ) : <Skeleton className="h-[320px] w-full" />}
        </Card>

        <Card>
          <CardHeader
            title="Error distribution"
            hint="How far predictions sat from the market price."
          />
          {bands.length ? (
            <div className="space-y-3.5">
              {bands.map(b => {
                const share = (b.count / bandTotal) * 100;
                return (
                  <div key={b.band}>
                    <div className="mb-1.5 flex justify-between text-xs">
                      <span className="text-ink-muted">{b.band}</span>
                      <span className="tabular text-ink-faint">
                        {share.toFixed(0)}% · {num(b.count)}
                      </span>
                    </div>
                    <div className="h-1.5 rounded-full bg-surface-3">
                      <div className="h-full rounded-full"
                           style={{ width: `${share}%`, background: bandTone(b.band) }} />
                    </div>
                  </div>
                );
              })}
            </div>
          ) : <Skeleton className="h-40 w-full" />}
        </Card>
      </div>

      <Card>
        <CardHeader title="How to read these numbers" />
        <div className="grid grid-cols-1 gap-x-8 gap-y-5 text-sm leading-relaxed text-ink-muted md:grid-cols-3">
          <div>
            <h3 className="mb-1.5 font-medium text-ink">Context beats output</h3>
            <p>
              Club prestige and international caps together carry more weight than
              goals and assists combined. Transfer fees price reputation and the
              standard a player has already been trusted at.
            </p>
          </div>
          <div>
            <h3 className="mb-1.5 font-medium text-ink">Error is not all noise</h3>
            <p>
              A mean error near 50% sounds poor until you note that market values
              are set by negotiation, are revised in round numbers, and move on
              news no dataset contains.
            </p>
          </div>
          <div>
            <h3 className="mb-1.5 font-medium text-ink">The blind spot</h3>
            <p>
              The model sees what a player has already done. It cannot see what a
              19-year-old is about to become, which is why the biggest
              disagreements cluster among the youngest players.
            </p>
          </div>
        </div>
      </Card>
    </>
  );
}
