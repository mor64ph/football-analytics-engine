import { useEffect, useRef, useState } from 'react';
import { api } from '../api/client';
import { Card, CardHeader, Field, PageHeader, Select, Skeleton } from '../components/ui';
import { competitionName, eur } from '../lib/format';
import type { PredictRequest } from '../types';

/** Median squad value per league, used as the default club-prestige input. */
const COMP_PRESTIGE: Record<string, number> = {
  PL: 18_000_000, PD: 14_000_000, SA: 10_000_000,
  BL1: 12_000_000, FL1: 8_000_000,
};

const DEFAULTS: PredictRequest = {
  age: 24,
  goals_p90: 0.45,
  assists_p90: 0.25,
  minutes_played: 2500,
  position: 'ATT',
  competition: 'PL',
  international_caps: 20,
  club_prestige_eur: 18_000_000,
};

/**
 * Bands for context, not flattery.
 *
 * The previous version rendered the figure at 72px with a coloured glow and a
 * label like "World Class", which reads as a game rather than a valuation tool.
 * The bands survive because a number is easier to judge against a scale, but
 * they are stated plainly.
 */
const BANDS = [
  { label: 'Squad player', max: 10e6 },
  { label: 'Regular starter', max: 30e6 },
  { label: 'Established quality', max: 60e6 },
  { label: 'Elite', max: 100e6 },
  { label: 'Generational', max: Infinity },
];

function Slider({
  label, min, max, step, value, onChange, format,
}: {
  label: string; min: number; max: number; step: number;
  value: number; onChange: (v: number) => void; format?: (v: number) => string;
}) {
  return (
    <div>
      <div className="mb-2 flex items-baseline justify-between">
        <span className="text-micro font-medium uppercase tracking-wider text-ink-faint">
          {label}
        </span>
        <span className="tabular text-sm font-medium text-ink">
          {format ? format(value) : value}
        </span>
      </div>
      <input
        type="range" min={min} max={max} step={step} value={value}
        onChange={e => onChange(Number(e.target.value))}
        aria-label={label}
      />
      <div className="mt-1 flex justify-between text-micro text-ink-faint">
        <span>{format ? format(min) : min}</span>
        <span>{format ? format(max) : max}</span>
      </div>
    </div>
  );
}

export default function Simulator() {
  const [params, setParams] = useState<PredictRequest>(DEFAULTS);
  const [value, setValue] = useState<number | null>(null);
  const [loading, setLoading] = useState(true);
  const [failed, setFailed] = useState(false);
  const debounce = useRef<ReturnType<typeof setTimeout> | null>(null);

  const set = <K extends keyof PredictRequest>(key: K) => (v: PredictRequest[K]) => {
    setParams(p => {
      const next = { ...p, [key]: v };
      // Club prestige is the strongest single feature, so it tracks the league
      // unless the user overrides it deliberately.
      if (key === 'competition') {
        next.club_prestige_eur = COMP_PRESTIGE[v as string] ?? p.club_prestige_eur;
      }
      return next;
    });
  };

  useEffect(() => {
    if (debounce.current) clearTimeout(debounce.current);
    debounce.current = setTimeout(() => {
      setLoading(true);
      api.predict(params)
        .then(r => { setValue(r.predicted_value_eur); setFailed(false); })
        .catch(() => setFailed(true))
        .finally(() => setLoading(false));
    }, 300);
    return () => { if (debounce.current) clearTimeout(debounce.current); };
  }, [params]);

  const band = value !== null ? BANDS.find(b => value < b.max) : null;

  return (
    <>
      <PageHeader
        title="Value Simulator"
        subtitle="Adjust a player's profile and see what the model would pay for him."
      />

      <div className="grid grid-cols-1 gap-6 lg:grid-cols-5">
        {/* Inputs */}
        <Card className="lg:col-span-3">
          <CardHeader title="Player profile" />

          <div className="mb-6 grid grid-cols-1 gap-4 sm:grid-cols-2">
            <Field label="Position">
              <Select value={params.position} onChange={set('position')}>
                <option value="GK">Goalkeeper</option>
                <option value="DEF">Defender</option>
                <option value="MID">Midfielder</option>
                <option value="ATT">Attacker</option>
              </Select>
            </Field>
            <Field label="League">
              <Select value={params.competition} onChange={set('competition')}>
                {Object.keys(COMP_PRESTIGE).map(c => (
                  <option key={c} value={c}>{competitionName(c)}</option>
                ))}
              </Select>
            </Field>
          </div>

          <div className="space-y-5">
            <Slider label="Age" min={16} max={38} step={1}
                    value={params.age} onChange={set('age')} />
            <Slider label="Goals per 90" min={0} max={1.2} step={0.05}
                    value={params.goals_p90} onChange={set('goals_p90')}
                    format={v => v.toFixed(2)} />
            <Slider label="Assists per 90" min={0} max={0.8} step={0.05}
                    value={params.assists_p90} onChange={set('assists_p90')}
                    format={v => v.toFixed(2)} />
            <Slider label="Minutes played" min={500} max={3500} step={100}
                    value={params.minutes_played} onChange={set('minutes_played')}
                    format={v => v.toLocaleString()} />
            <Slider label="International caps" min={0} max={120} step={1}
                    value={params.international_caps} onChange={set('international_caps')} />
            <Slider label="Club prestige" min={1e6} max={50e6} step={1e6}
                    value={params.club_prestige_eur} onChange={set('club_prestige_eur')}
                    format={v => eur(v)} />
          </div>
        </Card>

        {/* Output */}
        <div className="lg:col-span-2">
          <Card className="sticky top-8">
            <CardHeader title="Estimated value" />

            {failed ? (
              <p className="text-sm text-negative">Could not reach the model.</p>
            ) : (
              <>
                <div className="py-2">
                  {loading && value === null ? (
                    <Skeleton className="h-12 w-40" />
                  ) : (
                    <p className={`tabular text-4xl font-semibold tracking-tight text-ink
                                   transition-opacity ${loading ? 'opacity-40' : ''}`}>
                      {eur(value)}
                    </p>
                  )}
                  {band && !loading && (
                    <p className="mt-2 text-sm text-ink-muted">{band.label}</p>
                  )}
                </div>

                <dl className="mt-6 space-y-2 border-t border-line pt-4 text-xs">
                  {[
                    ['Position', params.position],
                    ['League', competitionName(params.competition)],
                    ['Age', params.age],
                    ['Output', `${params.goals_p90.toFixed(2)}G + ${params.assists_p90.toFixed(2)}A per 90`],
                    ['Minutes', params.minutes_played.toLocaleString()],
                    ['Caps', params.international_caps],
                  ].map(([k, v]) => (
                    <div key={String(k)} className="flex justify-between gap-4">
                      <dt className="text-ink-faint">{k}</dt>
                      <dd className="tabular text-ink-muted">{v}</dd>
                    </div>
                  ))}
                </dl>

                <p className="mt-5 border-t border-line pt-4 text-xs leading-relaxed text-ink-faint">
                  Club prestige and international caps carry the most weight in this
                  model — a player's own output matters less than where he plays and
                  who has already picked him.
                </p>
              </>
            )}
          </Card>
        </div>
      </div>
    </>
  );
}
