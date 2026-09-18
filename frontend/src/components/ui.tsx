import type { ReactNode } from 'react';

/**
 * The product's UI primitives.
 *
 * Before this existed, every page declared its own `panelStyle` object and its
 * own muted-text colour, so "a panel" meant seven slightly different things.
 * These components are the vocabulary the pages are written in; nothing below
 * this file should contain a raw colour or a bespoke border.
 */

const cx = (...parts: (string | false | null | undefined)[]) =>
  parts.filter(Boolean).join(' ');

/* ── Surface ─────────────────────────────────────────────────────────────── */

export function Card({
  children, className, padded = true,
}: { children: ReactNode; className?: string; padded?: boolean }) {
  return (
    <section
      className={cx(
        'rounded-lg border border-line bg-surface-1',
        padded && 'p-4 sm:p-5',
        className,
      )}
    >
      {children}
    </section>
  );
}

/**
 * Card heading with an optional explanatory line.
 *
 * The `hint` is not decoration. A chart that needs a paragraph to interpret is
 * a chart most readers will skip, and analysis products live or die on whether
 * the reader trusts they have understood the axis.
 */
export function CardHeader({
  title, hint, action,
}: { title: ReactNode; hint?: ReactNode; action?: ReactNode }) {
  return (
    <header className="mb-5 flex items-start justify-between gap-4">
      <div className="min-w-0">
        <h2 className="text-sm font-semibold tracking-tight text-ink">{title}</h2>
        {hint && <p className="mt-1 max-w-prose text-xs text-ink-faint">{hint}</p>}
      </div>
      {action && <div className="shrink-0">{action}</div>}
    </header>
  );
}

/* ── Page shell ──────────────────────────────────────────────────────────── */

export function PageHeader({
  title, subtitle, action,
}: { title: string; subtitle?: ReactNode; action?: ReactNode }) {
  return (
    // Wraps rather than squashes: on a narrow screen the action controls drop
    // below the title instead of compressing it to one word per line.
    <header className="flex flex-col gap-4 border-b border-line pb-5 sm:flex-row
                       sm:items-end sm:justify-between sm:gap-6 sm:pb-6">
      <div className="min-w-0">
        <h1 className="text-xl font-semibold tracking-tight text-ink">{title}</h1>
        {subtitle && <p className="mt-1.5 text-sm text-ink-muted">{subtitle}</p>}
      </div>
      {action && <div className="shrink-0">{action}</div>}
    </header>
  );
}

/* ── Data display ────────────────────────────────────────────────────────── */

type Tone = 'neutral' | 'positive' | 'negative' | 'caution' | 'info';

const toneText: Record<Tone, string> = {
  neutral:  'text-ink',
  positive: 'text-positive',
  negative: 'text-negative',
  caution:  'text-caution',
  info:     'text-info',
};

/**
 * A single figure with its label and context.
 *
 * Replaces the old KPICard, which took a pre-formatted string. That design let
 * callers write `${x?.toFixed(1)}%`, which silently renders "undefined%" before
 * data arrives - a bug that reached production. This takes the value and a
 * loading flag, so the not-yet state is the component's responsibility rather
 * than every caller's.
 */
export function Stat({
  label, value, detail, tone = 'neutral', loading,
}: {
  label: string;
  value: ReactNode;
  detail?: ReactNode;
  tone?: Tone;
  loading?: boolean;
}) {
  const pending = loading || value === null || value === undefined || value === '';

  return (
    <div className="rounded-lg border border-line bg-surface-1 p-4">
      <p className="text-micro font-medium uppercase tracking-wider text-ink-faint">
        {label}
      </p>
      {pending ? (
        <div className="skeleton mt-2.5 h-7 w-24" />
      ) : (
        <p className={cx('tabular mt-2 text-stat font-semibold tracking-tight', toneText[tone])}>
          {value}
        </p>
      )}
      {detail && !pending && (
        <p className="mt-1.5 truncate text-xs text-ink-muted" title={typeof detail === 'string' ? detail : undefined}>
          {detail}
        </p>
      )}
      {pending && <div className="skeleton mt-2 h-3 w-32" />}
    </div>
  );
}

export function Badge({
  children, tone = 'neutral',
}: { children: ReactNode; tone?: Tone }) {
  const tones: Record<Tone, string> = {
    neutral:  'border-line-strong text-ink-muted',
    positive: 'border-positive/30 text-positive',
    negative: 'border-negative/30 text-negative',
    caution:  'border-caution/30 text-caution',
    info:     'border-info/30 text-info',
  };
  return (
    <span className={cx(
      'inline-flex items-center rounded-sm border px-1.5 py-0.5 text-micro font-medium',
      tones[tone],
    )}>
      {children}
    </span>
  );
}

/** Signed delta. Colour carries the sign, so the arrow is not load-bearing. */
export function Delta({ value, format }: { value: number | null | undefined; format: (v: number) => string }) {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return <span className="text-ink-faint">—</span>;
  }
  const tone = value > 0 ? 'text-positive' : value < 0 ? 'text-negative' : 'text-ink-faint';
  return <span className={cx('tabular font-medium', tone)}>{format(value)}</span>;
}

/* ── States ──────────────────────────────────────────────────────────────── */

export function Skeleton({ className }: { className?: string }) {
  return <div className={cx('skeleton', className)} />;
}

export function EmptyState({
  title, hint, icon,
}: { title: string; hint?: ReactNode; icon?: ReactNode }) {
  return (
    <div className="flex flex-col items-center justify-center gap-2 px-6 py-12 text-center">
      {icon && <div className="text-ink-faint">{icon}</div>}
      <p className="text-sm font-medium text-ink-muted">{title}</p>
      {hint && <p className="max-w-sm text-xs text-ink-faint">{hint}</p>}
    </div>
  );
}

export function ErrorState({ message }: { message: string }) {
  return (
    <div className="rounded-md border border-negative/25 bg-negative/5 px-4 py-3 text-sm text-negative">
      {message}
    </div>
  );
}

/* ── Controls ────────────────────────────────────────────────────────────── */

export function Segmented<T extends string | number>({
  value, options, onChange,
}: {
  value: T;
  options: { value: T; label: string }[];
  onChange: (v: T) => void;
}) {
  return (
    <div role="tablist" className="inline-flex rounded-md border border-line bg-surface-2 p-0.5">
      {options.map(o => {
        const active = o.value === value;
        return (
          <button
            key={String(o.value)}
            role="tab"
            aria-selected={active}
            onClick={() => onChange(o.value)}
            className={cx(
              'rounded-sm px-2.5 py-1 text-xs font-medium transition-colors',
              active
                ? 'bg-surface-3 text-ink'
                : 'text-ink-faint hover:text-ink-muted',
            )}
          >
            {o.label}
          </button>
        );
      })}
    </div>
  );
}

export function Field({
  label, children, hint,
}: { label: string; children: ReactNode; hint?: string }) {
  return (
    <label className="block">
      <span className="mb-1.5 block text-micro font-medium uppercase tracking-wider text-ink-faint">
        {label}
      </span>
      {children}
      {hint && <span className="mt-1 block text-micro text-ink-faint">{hint}</span>}
    </label>
  );
}

const controlClass =
  'w-full rounded-md border border-line bg-surface-2 px-2.5 py-2 text-sm text-ink ' +
  'transition-colors hover:border-line-strong focus:border-brand focus:outline-none';

export function Select({
  value, onChange, children, disabled,
}: {
  value: string;
  onChange: (v: string) => void;
  children: ReactNode;
  disabled?: boolean;
}) {
  return (
    <select
      className={cx(controlClass, disabled && 'opacity-50')}
      value={value}
      disabled={disabled}
      onChange={e => onChange(e.target.value)}
    >
      {children}
    </select>
  );
}

export function Input({
  value, onChange, placeholder, type = 'text', step,
}: {
  value: string;
  onChange: (v: string) => void;
  placeholder?: string;
  type?: string;
  step?: string;
}) {
  return (
    <input
      className={controlClass}
      type={type}
      step={step}
      value={value}
      placeholder={placeholder}
      onChange={e => onChange(e.target.value)}
    />
  );
}

export function Button({
  children, onClick, disabled, variant = 'primary',
}: {
  children: ReactNode;
  onClick?: () => void;
  disabled?: boolean;
  variant?: 'primary' | 'ghost';
}) {
  const variants = {
    primary: 'bg-brand text-surface-0 hover:bg-brand-strong',
    ghost:   'border border-line text-ink-muted hover:border-line-strong hover:text-ink',
  };
  return (
    <button
      onClick={onClick}
      disabled={disabled}
      className={cx(
        'rounded-md px-3.5 py-2 text-sm font-medium transition-colors',
        'disabled:cursor-not-allowed disabled:opacity-40',
        variants[variant],
      )}
    >
      {children}
    </button>
  );
}

/* ── Table ───────────────────────────────────────────────────────────────── */

export function Table({ children }: { children: ReactNode }) {
  return (
    <div className="-mx-5 overflow-x-auto px-5">
      <table className="w-full border-collapse text-sm">{children}</table>
    </div>
  );
}

export function Th({
  children, align = 'left', className,
}: { children?: ReactNode; align?: 'left' | 'right'; className?: string }) {
  return (
    <th className={cx(
      'border-b border-line pb-2 text-micro font-medium uppercase tracking-wider text-ink-faint',
      align === 'right' ? 'text-right' : 'text-left',
      className,
    )}>
      {children}
    </th>
  );
}

export function Td({
  children, align = 'left', className,
}: { children?: ReactNode; align?: 'left' | 'right'; className?: string }) {
  return (
    <td className={cx(
      'border-b border-line/60 py-2.5',
      align === 'right' ? 'tabular text-right' : 'text-left',
      className,
    )}>
      {children}
    </td>
  );
}
