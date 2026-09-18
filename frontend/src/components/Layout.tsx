import { useEffect, useState, type ReactElement, type ReactNode } from 'react';
import { NavLink, useLocation } from 'react-router-dom';

/**
 * Application shell.
 *
 * Two changes of substance over the previous version. The navigation is
 * grouped rather than a flat list of seven items - a reader can now tell at a
 * glance that this product does two different jobs, valuation and match
 * prediction, which the old sidebar buried. And hover states are CSS rather
 * than onMouseEnter handlers writing to element.style, which broke keyboard
 * focus and left stale inline styles behind.
 */

const stroke = {
  fill: 'none',
  stroke: 'currentColor',
  strokeWidth: 1.5,
  strokeLinecap: 'round' as const,
  strokeLinejoin: 'round' as const,
};

const Icon = {
  overview: (
    <svg viewBox="0 0 16 16" className="size-4" {...stroke}>
      <rect x="2" y="2" width="5" height="5" rx="1" />
      <rect x="9" y="2" width="5" height="5" rx="1" />
      <rect x="2" y="9" width="5" height="5" rx="1" />
      <rect x="9" y="9" width="5" height="5" rx="1" />
    </svg>
  ),
  search: (
    <svg viewBox="0 0 16 16" className="size-4" {...stroke}>
      <circle cx="7" cy="7" r="4.5" />
      <path d="M10.5 10.5 14 14" />
    </svg>
  ),
  sliders: (
    <svg viewBox="0 0 16 16" className="size-4" {...stroke}>
      <path d="M2 4.5h12M2 11.5h12" />
      <circle cx="6" cy="4.5" r="1.8" />
      <circle cx="10" cy="11.5" r="1.8" />
    </svg>
  ),
  bars: (
    <svg viewBox="0 0 16 16" className="size-4" {...stroke}>
      <path d="M2.5 13V8M8 13V3.5M13.5 13v-7" />
    </svg>
  ),
  trend: (
    <svg viewBox="0 0 16 16" className="size-4" {...stroke}>
      <path d="M2 11.5 6 7l3 2.5L14 4" />
      <path d="M10.5 4H14v3.5" />
    </svg>
  ),
  pitch: (
    <svg viewBox="0 0 16 16" className="size-4" {...stroke}>
      <circle cx="8" cy="8" r="6" />
      <path d="M8 4.4 11 6.6l-1.15 3.5h-3.7L5 6.6z" />
    </svg>
  ),
  model: (
    <svg viewBox="0 0 16 16" className="size-4" {...stroke}>
      <circle cx="4" cy="4" r="1.8" />
      <circle cx="12" cy="8" r="1.8" />
      <circle cx="4" cy="12" r="1.8" />
      <path d="M5.6 4.8 10.4 7.2M5.6 11.2 10.4 8.8" />
    </svg>
  ),
};

const SECTIONS: { heading: string; links: { to: string; label: string; icon: ReactElement }[] }[] = [
  {
    heading: 'Market',
    links: [
      { to: '/',            label: 'Overview',         icon: Icon.overview },
      { to: '/players',     label: 'Player Search',    icon: Icon.search },
      { to: '/simulator',   label: 'Value Simulator',  icon: Icon.sliders },
      { to: '/leagues',     label: 'League Comparison', icon: Icon.bars },
      { to: '/leaderboard', label: 'Value Movers',     icon: Icon.trend },
    ],
  },
  {
    heading: 'Matches',
    links: [
      { to: '/matches', label: 'Match Predictor', icon: Icon.pitch },
    ],
  },
  {
    heading: 'Method',
    links: [
      { to: '/explainability', label: 'Model Insights', icon: Icon.model },
    ],
  },
];

export default function Layout({ children }: { children: ReactNode }) {
  const [navOpen, setNavOpen] = useState(false);
  const [slow, setSlow] = useState(false);
  const location = useLocation();

  useEffect(() => {
    const onSlow = () => setSlow(true);
    const onOk = () => setSlow(false);
    window.addEventListener('api-slow', onSlow);
    window.addEventListener('api-ok', onOk);
    return () => {
      window.removeEventListener('api-slow', onSlow);
      window.removeEventListener('api-ok', onOk);
    };
  }, []);

  // Navigating on a phone should dismiss the drawer; leaving it open would
  // cover the page the user just asked for.
  useEffect(() => { setNavOpen(false); }, [location.pathname]);

  // Escape closes it, which is what anyone who has used a drawer expects.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') setNavOpen(false); };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, []);

  return (
    <div className="flex min-h-screen bg-surface-0 text-ink">
      {/* Scrim: only rendered when the drawer is open, and only on small screens. */}
      {navOpen && (
        <div
          className="fixed inset-0 z-30 bg-black/60 lg:hidden"
          onClick={() => setNavOpen(false)}
          aria-hidden
        />
      )}

      <aside
        className={[
          'fixed inset-y-0 left-0 z-40 flex h-screen w-60 shrink-0 flex-col',
          'border-r border-line bg-surface-1 transition-transform duration-200',
          navOpen ? 'translate-x-0' : '-translate-x-full',
          // From lg up it is a permanent column, not a drawer.
          'lg:sticky lg:top-0 lg:translate-x-0',
        ].join(' ')}
      >
        {/* Wordmark */}
        <div className="flex items-center gap-2.5 px-5 py-5">
          <svg viewBox="0 0 24 24" className="size-7 shrink-0" aria-hidden>
            <rect width="24" height="24" rx="6" className="fill-brand-dim" />
            <path d="M6 16.5V11m6 5.5V7m6 9.5v-3.5"
                  stroke="var(--color-brand)" strokeWidth="2" strokeLinecap="round" fill="none" />
          </svg>
          <div className="leading-tight">
            <div className="text-sm font-semibold tracking-tight">ScoutIQ</div>
            <div className="text-micro text-ink-faint">Football Intelligence</div>
          </div>
        </div>

        <nav className="flex-1 overflow-y-auto px-3 pb-4">
          {SECTIONS.map(section => (
            <div key={section.heading} className="mb-5">
              <p className="mb-1.5 px-2 text-micro font-medium uppercase tracking-wider text-ink-faint">
                {section.heading}
              </p>
              <ul className="space-y-0.5">
                {section.links.map(({ to, label, icon }) => (
                  <li key={to}>
                    <NavLink
                      to={to}
                      end={to === '/'}
                      className={({ isActive }) =>
                        [
                          'group relative flex items-center gap-2.5 rounded-md px-2 py-1.5',
                          'text-sm transition-colors',
                          isActive
                            ? 'bg-surface-3 font-medium text-ink'
                            : 'text-ink-muted hover:bg-surface-2 hover:text-ink',
                        ].join(' ')
                      }
                    >
                      {({ isActive }) => (
                        <>
                          {/* Active marker reads even for someone who cannot
                              distinguish the background shade. */}
                          <span
                            aria-hidden
                            className={[
                              'absolute left-0 h-4 w-0.5 rounded-r-full transition-colors',
                              isActive ? 'bg-brand' : 'bg-transparent',
                            ].join(' ')}
                          />
                          <span className={isActive ? 'text-brand' : 'text-ink-faint group-hover:text-ink-muted'}>
                            {icon}
                          </span>
                          {label}
                        </>
                      )}
                    </NavLink>
                  </li>
                ))}
              </ul>
            </div>
          ))}
        </nav>

        <footer className="space-y-1 border-t border-line px-5 py-4">
          <p className="text-micro text-ink-faint">
            Premier League · LaLiga · Serie A · Bundesliga · Ligue 1
          </p>
          <p className="text-micro text-ink-faint/70">Data refreshed daily</p>
        </footer>
      </aside>

      <main className="min-w-0 flex-1">
        {/* Mobile bar: the only way to reach navigation below lg. */}
        <div className="sticky top-0 z-20 flex items-center gap-3 border-b border-line
                        bg-surface-0/95 px-4 py-3 backdrop-blur lg:hidden">
          <button
            onClick={() => setNavOpen(true)}
            aria-label="Open navigation"
            aria-expanded={navOpen}
            className="rounded-md border border-line p-1.5 text-ink-muted
                       transition-colors hover:border-line-strong hover:text-ink"
          >
            <svg viewBox="0 0 16 16" className="size-4" fill="none" stroke="currentColor"
                 strokeWidth="1.5" strokeLinecap="round">
              <path d="M2 4h12M2 8h12M2 12h12" />
            </svg>
          </button>
          <span className="text-sm font-semibold tracking-tight">ScoutIQ</span>
        </div>

        <div className="mx-auto max-w-[1400px] space-y-6 px-4 py-6 sm:px-6 lg:space-y-8 lg:px-8 lg:py-8">
          {children}
        </div>
      </main>

      {/* Connecting pill — appears only during slow API calls (cold start).
          Not "server waking up" copy; looks like a first-party status indicator. */}
      {slow && (
        <div className="fixed bottom-5 right-5 z-50 flex items-center gap-2 rounded-full
                        border border-line-strong bg-surface-2 px-3.5 py-2
                        text-micro text-ink-muted shadow-lg shadow-black/40">
          <span className="size-1.5 shrink-0 animate-pulse rounded-full bg-caution" />
          Connecting…
        </div>
      )}
    </div>
  );
}
