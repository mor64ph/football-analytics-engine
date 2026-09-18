import { useEffect, useState } from 'react';
import { api } from '../api/client';
import {
  Badge, Button, Card, CardHeader, EmptyState, ErrorState, Field,
  PageHeader, Segmented, Select, Skeleton, Table, Td, Th,
} from '../components/ui';
import { CHART } from '../lib/chart';
import { competitionName, prob } from '../lib/format';
import type { Fixture, MatchPrediction, TeamRating } from '../types';

const OUTCOME_COLORS = [CHART.series[0], CHART.axis, CHART.series[1]];

/* ── Probability display ─────────────────────────────────────────────────── */

function OutcomeBar({ p, compact }: { p: MatchPrediction; compact?: boolean }) {
  const seg = [
    { v: p.probabilities.home_win, c: OUTCOME_COLORS[0], label: p.home_team },
    { v: p.probabilities.draw,     c: OUTCOME_COLORS[1], label: 'Draw' },
    { v: p.probabilities.away_win, c: OUTCOME_COLORS[2], label: p.away_team },
  ];

  return (
    <div className="space-y-3">
      <div className={`flex overflow-hidden rounded-md ${compact ? 'h-1.5' : 'h-9'}`}>
        {seg.map((s, i) => (
          <div
            key={i}
            style={{ width: `${s.v * 100}%`, background: s.c }}
            className="flex items-center justify-center text-xs font-semibold text-surface-0"
          >
            {!compact && s.v > 0.1 ? prob(s.v, 0) : null}
          </div>
        ))}
      </div>
      {!compact && (
        <div className="grid grid-cols-3 gap-3">
          {seg.map((s, i) => (
            <div key={i} className="min-w-0">
              <div className="truncate text-micro text-ink-faint">{s.label}</div>
              <div className="tabular text-base font-semibold" style={{ color: s.c }}>
                {prob(s.v)}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

function Metric({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-md border border-line bg-surface-2 px-3 py-2.5">
      <div className="text-micro text-ink-faint">{label}</div>
      <div className="tabular mt-0.5 text-sm font-semibold text-ink">{value}</div>
    </div>
  );
}

/** Joint scoreline distribution. Colour encodes the outcome, opacity the probability. */
function ScoreGrid({ p }: { p: MatchPrediction }) {
  const grid = p.scoreline_grid;
  const max = Math.max(...grid.flat());

  return (
    <div>
      <div className="inline-block">
        <div className="flex">
          <div className="w-6" />
          {[0, 1, 2, 3, 4, 5].map(a => (
            <div key={a} className="w-10 pb-1 text-center text-micro text-ink-faint">{a}</div>
          ))}
        </div>
        {grid.map((row, h) => (
          <div key={h} className="flex items-center">
            <div className="w-6 text-center text-micro text-ink-faint">{h}</div>
            {row.map((v, a) => {
              const intensity = max > 0 ? v / max : 0;
              const colour = h > a ? OUTCOME_COLORS[0] : h === a ? OUTCOME_COLORS[1] : OUTCOME_COLORS[2];
              return (
                <div
                  key={a}
                  title={`${h}–${a}: ${prob(v)}`}
                  className="m-px flex h-8 w-10 items-center justify-center rounded-sm text-micro font-medium"
                  style={{
                    background: colour,
                    opacity: Math.max(intensity * 0.9, 0.05),
                    color: intensity > 0.5 ? 'var(--color-surface-0)' : 'var(--color-ink)',
                  }}
                >
                  {v >= 0.012 ? (v * 100).toFixed(0) : ''}
                </div>
              );
            })}
          </div>
        ))}
      </div>
      <p className="mt-3 text-micro text-ink-faint">
        Rows are {p.home_team} goals, columns {p.away_team}. Values are the chance of that exact score.
      </p>
    </div>
  );
}

/* ── Fixtures ────────────────────────────────────────────────────────────── */

function FixtureRow({ f, onPick }: { f: Fixture; onPick: (f: Fixture) => void }) {
  const p = f.prediction;
  const time = new Date(f.utc_date).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });

  return (
    <button
      onClick={() => p && onPick(f)}
      disabled={!p}
      className="w-full border-b border-line/60 px-5 py-3 text-left transition-colors last:border-0
                 enabled:hover:bg-surface-2 disabled:cursor-default"
    >
      <div className="flex items-center gap-3">
        <span className="tabular w-10 shrink-0 text-micro text-ink-faint">{time}</span>
        <span className="w-16 shrink-0 text-micro text-ink-faint">{f.competition}</span>
        <span className="min-w-0 flex-1 truncate text-sm text-ink">
          {f.home_team} <span className="text-ink-faint">v</span> {f.away_team}
        </span>
        {p ? (
          <span className="tabular shrink-0 text-micro text-ink-muted">
            {prob(p.probabilities.home_win, 0)} · {prob(p.probabilities.draw, 0)} · {prob(p.probabilities.away_win, 0)}
          </span>
        ) : (
          <span className="shrink-0 text-micro text-ink-faint">unrated</span>
        )}
      </div>
      {p && (
        <div className="mt-2 flex items-center gap-3">
          <span className="w-10 shrink-0" />
          <span className="w-16 shrink-0" />
          <div className="flex-1"><OutcomeBar p={p} compact /></div>
          <span className="tabular w-20 shrink-0 text-right text-micro text-ink-faint">
            xG {p.expected_goals.home.toFixed(1)}–{p.expected_goals.away.toFixed(1)}
          </span>
        </div>
      )}
    </button>
  );
}

/* ── Page ────────────────────────────────────────────────────────────────── */

export default function MatchPredictor() {
  const [comps, setComps] = useState<{ code: string; name: string }[]>([]);
  const [comp, setComp] = useState('PL');
  const [teams, setTeams] = useState<string[]>([]);
  const [home, setHome] = useState('');
  const [away, setAway] = useState('');
  const [pred, setPred] = useState<MatchPrediction | null>(null);
  const [ratings, setRatings] = useState<TeamRating[]>([]);
  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  const [fixtures, setFixtures] = useState<Fixture[]>([]);
  const [fixDays, setFixDays] = useState(7);
  const [fixLoading, setFixLoading] = useState(true);
  const [accuracy, setAccuracy] = useState<Awaited<ReturnType<typeof api.getMatchAccuracy>> | null>(null);

  useEffect(() => {
    api.getMatchCompetitions()
      .then(r => setComps(r.data))
      .catch(() => setErr('Match model unavailable.'));
    api.getMatchAccuracy().then(setAccuracy).catch(() => {});
  }, []);

  useEffect(() => {
    if (!comp) return;
    api.getMatchTeams(comp).then(r => {
      setTeams(r.data);
      setHome(r.data[0] ?? '');
      setAway(r.data[1] ?? '');
    }).catch(() => {});
    api.getMatchRatings({ competition: comp, limit: 20 })
      .then(r => setRatings(r.data)).catch(() => {});
  }, [comp]);

  useEffect(() => {
    setFixLoading(true);
    api.getFixtures(fixDays)
      .then(r => setFixtures(r.data ?? []))
      .catch(() => setFixtures([]))
      .finally(() => setFixLoading(false));
  }, [fixDays]);

  const run = async () => {
    if (!home || !away || home === away) return;
    setLoading(true); setErr(null);
    try {
      setPred(await api.predictMatch({ home_team: home, away_team: away, competition: comp }));
    } catch {
      setErr('Prediction failed.');
    } finally {
      setLoading(false);
    }
  };

  const pickFixture = (f: Fixture) => {
    if (!f.prediction) return;
    setPred(f.prediction);
    setComp(f.prediction.competition);
    setHome(f.prediction.home_team);
    setAway(f.prediction.away_team);
    window.scrollTo({ top: 0, behavior: 'smooth' });
  };

  const byDay = fixtures.reduce<Record<string, Fixture[]>>((acc, f) => {
    const d = new Date(f.utc_date).toLocaleDateString(undefined, {
      weekday: 'long', day: 'numeric', month: 'short',
    });
    (acc[d] ??= []).push(f);
    return acc;
  }, {});

  const priced = fixtures.filter(f => f.predicted).length;

  return (
    <>
      <PageHeader
        title="Match Predictor"
        subtitle="Outcome probabilities from team ratings; scorelines from a goal-scoring model."
        action={
          <Segmented
            value={fixDays}
            onChange={setFixDays}
            options={[{ value: 3, label: '3d' }, { value: 7, label: '7d' }, { value: 14, label: '14d' }]}
          />
        }
      />

      {err && <ErrorState message={err} />}

      {/* Fixture picker */}
      <Card>
        <CardHeader title="Price a fixture" />
        <div className="grid grid-cols-1 items-end gap-4 md:grid-cols-4">
          <Field label="Competition">
            <Select value={comp} onChange={setComp}>
              {comps.map(c => <option key={c.code} value={c.code}>{c.name}</option>)}
            </Select>
          </Field>
          <Field label="Home">
            <Select value={home} onChange={setHome}>
              {teams.map(t => <option key={t} value={t}>{t}</option>)}
            </Select>
          </Field>
          <Field label="Away">
            <Select value={away} onChange={setAway}>
              {teams.map(t => <option key={t} value={t}>{t}</option>)}
            </Select>
          </Field>
          <Button onClick={run} disabled={loading || !home || !away || home === away}>
            {loading ? 'Predicting…' : 'Predict'}
          </Button>
        </div>
      </Card>

      {/* Result */}
      {pred && (
        <>
          <Card>
            <CardHeader
              title={<>{pred.home_team} <span className="text-ink-faint">v</span> {pred.away_team}</>}
              action={<Badge>{competitionName(pred.competition)}</Badge>}
            />
            <OutcomeBar p={pred} />

            <div className="mt-5 grid grid-cols-2 gap-3 lg:grid-cols-4">
              <Metric label="Elo" value={`${pred.elo.home.toFixed(0)} – ${pred.elo.away.toFixed(0)}`} />
              <Metric label="Expected goals"
                      value={`${pred.expected_goals.home.toFixed(2)} – ${pred.expected_goals.away.toFixed(2)}`} />
              <Metric label="Over 2.5 goals" value={prob(pred.goals_markets.over_2_5)} />
              <Metric label="Both teams score" value={prob(pred.goals_markets.btts)} />
              <Metric label="Over 1.5" value={prob(pred.goals_markets.over_1_5)} />
              <Metric label="Over 3.5" value={prob(pred.goals_markets.over_3_5)} />
              <Metric label={`${pred.home_team} clean sheet`} value={prob(pred.goals_markets.home_clean_sheet)} />
              <Metric label={`${pred.away_team} clean sheet`} value={prob(pred.goals_markets.away_clean_sheet)} />
            </div>
          </Card>

          <div className="grid grid-cols-1 gap-6 xl:grid-cols-2">
            <Card>
              <CardHeader title="Most likely scorelines" />
              <ul className="space-y-2">
                {pred.top_scores.slice(0, 8).map(s => (
                  <li key={s.score} className="flex items-center gap-3">
                    <span className="tabular w-10 text-sm text-ink">{s.score.replace('-', '–')}</span>
                    <div className="h-1.5 flex-1 rounded-full bg-surface-3">
                      <div className="h-full rounded-full bg-brand"
                           style={{ width: `${(s.probability / pred.top_scores[0].probability) * 100}%` }} />
                    </div>
                    <span className="tabular w-12 text-right text-micro text-ink-muted">
                      {prob(s.probability)}
                    </span>
                  </li>
                ))}
              </ul>
            </Card>

            <Card>
              <CardHeader title="Scoreline distribution" />
              <ScoreGrid p={pred} />
            </Card>
          </div>
        </>
      )}

      {/* Upcoming fixtures */}
      <Card padded={false}>
        <div className="px-5 pt-5">
          <CardHeader
            title="Upcoming fixtures"
            hint={fixLoading ? 'Loading…' : `${priced} of ${fixtures.length} priced — select one for full detail`}
          />
        </div>
        {fixLoading ? (
          <div className="space-y-2 px-5 pb-5">
            {Array.from({ length: 5 }).map((_, i) => <Skeleton key={i} className="h-12 w-full" />)}
          </div>
        ) : fixtures.length === 0 ? (
          <EmptyState title="No fixtures scheduled"
                      hint={`Nothing in the next ${fixDays} days across the five leagues.`} />
        ) : (
          <div className="max-h-[32rem] overflow-y-auto">
            {Object.entries(byDay).map(([day, list]) => (
              <div key={day}>
                <div className="sticky top-0 z-10 border-y border-line bg-surface-2 px-5 py-1.5
                                text-micro font-medium uppercase tracking-wider text-ink-faint">
                  {day}
                </div>
                {list.map((f, i) => <FixtureRow key={`${f.utc_date}-${i}`} f={f} onPick={pickFixture} />)}
              </div>
            ))}
          </div>
        )}
      </Card>

      {/* Live accuracy. The backtest figure is a claim about the future made
          from the past; this is the only number that checks it against matches
          the model had not seen when it spoke. */}
      {accuracy && (
        <Card>
          <CardHeader
            title="Live accuracy"
            hint="Scored against fixtures predicted before kick-off, not against the training data."
          />
          {accuracy.scored === 0 ? (
            <p className="text-sm text-ink-muted">
              {accuracy.pending} prediction{accuracy.pending === 1 ? '' : 's'} logged
              and awaiting results. Figures appear once those fixtures have been played.
            </p>
          ) : (
            <>
              <div className="grid grid-cols-2 gap-4 lg:grid-cols-4">
                <Metric label="Matches scored" value={String(accuracy.scored)} />
                <Metric label="Log loss" value={accuracy.log_loss?.toFixed(4) ?? '—'} />
                <Metric label="Backtest said" value={accuracy.backtest_log_loss?.toFixed(4) ?? '—'} />
                <Metric label="Correct outcome"
                        value={accuracy.accuracy !== undefined ? prob(accuracy.accuracy) : '—'} />
              </div>
              {accuracy.scored < 100 && (
                <p className="mt-4 border-t border-line pt-4 text-xs text-ink-faint">
                  Fewer than 100 matches. The uncertainty on this figure is still wider
                  than the gap it is being compared against, so it is not yet a verdict.
                </p>
              )}
            </>
          )}
        </Card>
      )}

      {/* Power rankings */}
      {ratings.length > 0 && (
        <Card>
          <CardHeader
            title="Power rankings"
            hint="Elo measures overall strength. Attack and defence come from the goal model — higher is better for both."
            action={<Badge>{competitionName(comp)}</Badge>}
          />
          <Table>
            <thead>
              <tr>
                <Th className="w-8">#</Th>
                <Th>Team</Th>
                <Th align="right">Elo</Th>
                <Th align="right">Attack</Th>
                <Th align="right">Defence</Th>
              </tr>
            </thead>
            <tbody>
              {ratings.map((r, i) => (
                <tr key={r.team} className="transition-colors hover:bg-surface-2">
                  <Td className="text-ink-faint">{i + 1}</Td>
                  <Td>{r.team}</Td>
                  <Td align="right" className="font-medium text-ink">{r.elo.toFixed(0)}</Td>
                  <Td align="right" className="text-ink-muted">{r.attack?.toFixed(2) ?? '—'}</Td>
                  <Td align="right" className="text-ink-muted">{r.defence?.toFixed(2) ?? '—'}</Td>
                </tr>
              ))}
            </tbody>
          </Table>
        </Card>
      )}
    </>
  );
}
