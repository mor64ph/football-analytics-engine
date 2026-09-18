import type { ReactNode } from 'react';

/**
 * Shared chart styling.
 *
 * Every page previously declared its own `tooltipStyle` and repeated the same
 * axis tick config inline on each `<XAxis>`. The values drifted, so tooltips
 * and gridlines looked subtly different from one chart to the next. These
 * constants are the single definition; charts spread them in.
 */

const css = (name: string) =>
  getComputedStyle(document.documentElement).getPropertyValue(name).trim();

/** Read a design token at runtime so charts cannot drift from the CSS. */
export const token = (name: string, fallback: string) => {
  try {
    return css(name) || fallback;
  } catch {
    return fallback;
  }
};

export const CHART = {
  grid: '#21262e',
  axis: '#6b7280',
  series: ['#3fb98a', '#5b8dd9', '#d9a13b', '#a884d4', '#e06c65'],
  positive: '#3fb98a',
  negative: '#e06c65',
  caution: '#d9a13b',
} as const;

export const axisProps = {
  tick: { fill: CHART.axis, fontSize: 11 },
  tickLine: false,
  axisLine: { stroke: CHART.grid },
} as const;

export const gridProps = {
  stroke: CHART.grid,
  strokeDasharray: '0',
  vertical: false,
} as const;

/**
 * Tooltip container.
 *
 * recharts' default tooltip renders a bordered white box that has to be
 * restyled at every use site. This wraps content in the product's own surface
 * so charts stop looking like a third-party widget embedded in the page.
 */
export function ChartTooltip({
  title, rows,
}: {
  title?: ReactNode;
  rows: { label: string; value: ReactNode; color?: string }[];
}) {
  return (
    <div className="rounded-md border border-line-strong bg-surface-2 px-3 py-2 shadow-lg">
      {title && (
        <div className="mb-1.5 text-xs font-medium text-ink">{title}</div>
      )}
      <div className="space-y-1">
        {rows.map((r, i) => (
          <div key={i} className="flex items-center gap-2 text-micro">
            {r.color && (
              <span className="size-2 shrink-0 rounded-full"
                    style={{ background: r.color }} />
            )}
            <span className="text-ink-faint">{r.label}</span>
            <span className="tabular ml-auto font-medium text-ink">{r.value}</span>
          </div>
        ))}
      </div>
    </div>
  );
}
