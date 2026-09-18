import { useEffect, useState } from 'react';
import {
  Area, AreaChart, Bar, BarChart, CartesianGrid, Cell,
  ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis,
} from 'recharts';
import { api } from '../api/client';
import { getLocal, setLocal } from '../lib/cache';
import {
  Card, CardHeader, PageHeader, Stat, Skeleton, EmptyState, Badge,
} from '../components/ui';
import { CHART, ChartTooltip, axisProps, gridProps } from '../lib/chart';
import { competitionName, eur, eurM, num } from '../lib/format';
import type { KPIs, MarketInsights, ValuedPlayer } from '../types';

const SLOT_TONE: Record<string, string> = {
  GK: CHART.series[3], DEF: CHART.series[2], MID: CHART.series[1], ATT: CHART.series[0],
};

function XIRow({ p, max }: { p: ValuedPlayer; max: number }) {
  const colour = SLOT_TONE[p.slot ?? ''] ?? CHART.axis;
  return (
    <li className="flex items-center gap-3 border-b border-line/60 py-2.5 last:border-0">
      <span
        className="w-9 shrink-0 rounded-sm py-0.5 text-center text-micro font-semibold"
        style={{ background: `${colour}1f`, color: colour }}
      >
        {p.slot}
      </span>

      <div className="min-w-0 flex-1">
        <div className="truncate text-sm text-ink">{p.player_name}</div>
        <div className="text-micro text-ink-faint">
          {competitionName(p.competition)} · age {p.age} · {eur(p.market_value_eur)} market
        </div>
      </div>

      <div className="hidden w-20 shrink-0 sm:block">
        <div className="h-1 rounded-full bg-surface-3">
          <div className="h-full rounded-full bg-positive"
               style={{ width: `${Math.max((p.gap_eur / max) * 100, 4)}%` }} />
        </div>
      </div>

      <span className="tabular w-16 shrink-0 text-right text-sm font-medium text-positive">
        {eur(p.gap_eur, { sign: true })}
      </span>
    </li>
  );
}

export default function Overview() {
  // Initialise from cache so returning users see data on the first paint
  // rather than skeletons. The API calls below will update these values when
  // fresh data arrives.
  const [kpis, setKpis] = useState<KPIs | null>(() => getLocal('kpis'));
  const [mi, setMi] = useState<MarketInsights | null>(() => getLocal('market-insights'));
  const [failed, setFailed] = useState(false);

  useEffect(() => {
    api.getKPIs()
      .then(d => { setKpis(d); setLocal('kpis', d); })
      .catch(() => setFailed(true));
    api.getMarketInsights()
      .then(d => { setMi(d); setLocal('market-insights', d); })
      .catch(() => setFailed(true));
  }, []);

  const under = mi?.most_undervalued;
  const over = mi?.most_overvalued;
  const maxGap = Math.max(...(mi?.undervalued_xi.map(p => p.gap_eur) ?? [1]), 1);

  const leagues = (mi?.leagues ?? []).map(l => ({
    name: competitionName(l.competition),
    median: l.median,
    p75: l.p75,
    n: l.n,
  }));

  const mispricing = (mi?.mispricing_by_age ?? []).map(m => ({
    band: m.band, gap: m.median_gap_eur, n: m.n,
  }));

  return (
    <>
      <PageHeader
        title="Market Overview"
        subtitle={
          mi?.available && kpis
            ? `Where the model disagrees with the transfer market — ${num(mi.total_players)} players across ${kpis.competitions.length} leagues`
            : 'Loading market summary…'
        }
        action={mi?.available
          ? <Badge tone="neutral">{mi.season}/{String((mi.season + 1) % 100).padStart(2, '0')} season</Badge>
          : undefined}
      />

      {failed && !mi && !kpis && (
        <EmptyState
          title="Couldn't reach the API"
          hint="The service may be starting up. Refresh in a moment."
        />
      )}

      {/* Headline findings — about the market, not about model fit. */}
      <div className="grid grid-cols-1 gap-4 md:grid-cols-2 xl:grid-cols-4">
        <Stat
          label="Most undervalued" tone="positive"
          value={under ? eur(under.gap_eur, { sign: true }) : null}
          detail={under ? `${under.player_name} · ${eur(under.market_value_eur)} market price` : undefined}
        />
        <Stat
          label="Most overvalued" tone="negative"
          value={over ? eur(over.gap_eur, { sign: true }) : null}
          detail={over ? `${over.player_name} · ${eur(over.market_value_eur)} market price` : undefined}
        />
        <Stat
          label="Value tracked"
          value={mi?.available ? eur(mi.total_value_eur) : null}
          detail={mi?.available ? `${num(mi.total_players)} players, current season` : undefined}
        />
        <Stat
          label="Peak valuation age"
          value={mi?.peak_age ?? null}
          detail="Age at which median value is highest"
        />
      </div>

      {/* The value curve */}
      <Card>
        <CardHeader
          title="The value curve"
          hint="Median market value at each age. Clubs pay for the years they expect to get, not the years already played — which is why age is among the model's strongest features."
        />
        {mi?.age_curve.length ? (
          <ResponsiveContainer width="100%" height={280}>
            {/* Top margin leaves room for the peak marker's label, which
                recharts draws above the plot area and would otherwise clip. */}
            <AreaChart data={mi.age_curve} margin={{ top: 24, right: 12, left: 4, bottom: 0 }}>
              <defs>
                <linearGradient id="curve" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor={CHART.positive} stopOpacity={0.28} />
                  <stop offset="100%" stopColor={CHART.positive} stopOpacity={0} />
                </linearGradient>
              </defs>
              <CartesianGrid {...gridProps} />
              <XAxis dataKey="age" {...axisProps} />
              <YAxis tickFormatter={eurM} width={52} {...axisProps} />
              <Tooltip
                cursor={{ stroke: CHART.grid }}
                content={({ active, payload, label }) =>
                  active && payload?.length ? (
                    <ChartTooltip
                      title={`Age ${label}`}
                      rows={[
                        { label: 'Median value', value: eur(payload[0].payload.median_value_eur), color: CHART.positive },
                        { label: '75th percentile', value: eur(payload[0].payload.p75_value_eur) },
                        { label: 'Players', value: num(payload[0].payload.n) },
                      ]}
                    />
                  ) : null
                }
              />
              {mi.peak_age && (
                <ReferenceLine
                  x={mi.peak_age}
                  stroke={CHART.positive}
                  strokeDasharray="3 3"
                  label={{
                    value: `peak · ${mi.peak_age}`,
                    fill: CHART.positive,
                    fontSize: 11,
                    position: 'top',
                    offset: 8,
                  }}
                />
              )}
              <Area type="monotone" dataKey="median_value_eur"
                    stroke={CHART.positive} strokeWidth={2} fill="url(#curve)" />
            </AreaChart>
          </ResponsiveContainer>
        ) : (
          <Skeleton className="h-[280px] w-full" />
        )}
      </Card>

      <div className="grid grid-cols-1 gap-6 xl:grid-cols-2">
        {/* Undervalued XI */}
        <Card>
          <CardHeader
            title="The undervalued XI"
            hint="Largest valuation gap at each position, arranged into a team. Positive means the model rates a player above his market price."
          />
          {mi?.undervalued_xi.length ? (
            <ul>
              {mi.undervalued_xi.map((p, i) => <XIRow key={i} p={p} max={maxGap} />)}
            </ul>
          ) : (
            <div className="space-y-2">
              {Array.from({ length: 6 }).map((_, i) => <Skeleton key={i} className="h-10 w-full" />)}
            </div>
          )}
        </Card>

        {/* League distribution */}
        <Card>
          <CardHeader
            title="Where the money is"
            hint="Median and upper-quartile player value by league. A wide gap between the two marks a top-heavy division."
          />
          {leagues.length ? (
            <>
              <ResponsiveContainer width="100%" height={240}>
                <BarChart data={leagues} layout="vertical"
                          margin={{ top: 0, right: 12, left: 8, bottom: 0 }}>
                  <CartesianGrid {...gridProps} vertical horizontal={false} />
                  <XAxis type="number" tickFormatter={eurM} {...axisProps} />
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
                            { label: 'Median', value: eur(payload[0].payload.median), color: CHART.series[0] },
                            { label: '75th pct', value: eur(payload[0].payload.p75), color: CHART.series[1] },
                            { label: 'Players', value: num(payload[0].payload.n) },
                          ]}
                        />
                      ) : null
                    }
                  />
                  <Bar dataKey="median" fill={CHART.series[0]} radius={[0, 3, 3, 0]} barSize={10} />
                  <Bar dataKey="p75" fill={CHART.series[1]} radius={[0, 3, 3, 0]} barSize={10} />
                </BarChart>
              </ResponsiveContainer>
              <div className="mt-2 flex justify-center gap-4 text-micro text-ink-faint">
                <span className="flex items-center gap-1.5">
                  <span className="size-2 rounded-full" style={{ background: CHART.series[0] }} />Median
                </span>
                <span className="flex items-center gap-1.5">
                  <span className="size-2 rounded-full" style={{ background: CHART.series[1] }} />75th percentile
                </span>
              </div>
            </>
          ) : (
            <Skeleton className="h-[240px] w-full" />
          )}
        </Card>
      </div>

      {/* Mispricing by age */}
      <Card>
        <CardHeader
          title="Where the model and the market disagree"
          hint="Median gap by age band. Negative means clubs pay more than results alone can justify."
        />
        {mispricing.length ? (
          <>
            <ResponsiveContainer width="100%" height={200}>
              <BarChart data={mispricing} margin={{ top: 8, right: 8, left: 4, bottom: 0 }}>
                <CartesianGrid {...gridProps} />
                <XAxis dataKey="band" {...axisProps} />
                <YAxis tickFormatter={v => `€${(v / 1e6).toFixed(1)}M`} width={56} {...axisProps} />
                <Tooltip
                  cursor={{ fill: 'rgb(255 255 255 / 0.03)' }}
                  content={({ active, payload, label }) =>
                    active && payload?.length ? (
                      <ChartTooltip
                        title={`Age ${label}`}
                        rows={[
                          { label: 'Median gap', value: eur(payload[0].payload.gap, { sign: true }) },
                          { label: 'Players', value: num(payload[0].payload.n) },
                        ]}
                      />
                    ) : null
                  }
                />
                <ReferenceLine y={0} stroke={CHART.axis} />
                <Bar dataKey="gap" radius={[2, 2, 0, 0]} barSize={44}>
                  {mispricing.map((d, i) => (
                    <Cell key={i} fill={d.gap >= 0 ? CHART.positive : CHART.caution} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
            <p className="mt-4 max-w-prose border-t border-line pt-4 text-xs leading-relaxed text-ink-muted">
              The youngest band runs consistently negative. Clubs pay a premium for players
              whose best years are ahead of them, and a model trained on what has already
              happened cannot see that. It is a limit of the method rather than a fault in it —
              and it is why the most &ldquo;overvalued&rdquo; player in the league is usually
              the most exciting teenager in it.
            </p>
          </>
        ) : (
          <Skeleton className="h-[200px] w-full" />
        )}
      </Card>
    </>
  );
}
