import { Component, type ErrorInfo, type ReactNode } from 'react';

/**
 * Stops one broken component taking the whole application down.
 *
 * React unmounts the entire tree when a render throws and nothing catches it,
 * so a single bad array index anywhere produced a white page with no route, no
 * navigation and no way back except a manual reload. That is a poor failure
 * for any app and a particularly poor one for something people open from a
 * shared link.
 *
 * Must be a class: there is still no hook equivalent of componentDidCatch.
 */

interface Props { children: ReactNode }
interface State { error: Error | null }

export default class ErrorBoundary extends Component<Props, State> {
  state: State = { error: null };

  static getDerivedStateFromError(error: Error): State {
    return { error };
  }

  componentDidCatch(error: Error, info: ErrorInfo) {
    // Kept to the console rather than sent anywhere: there is no error
    // reporting service wired up, and pretending otherwise would be worse
    // than saying so.
    console.error('Unhandled render error:', error, info.componentStack);
  }

  render() {
    const { error } = this.state;
    if (!error) return this.props.children;

    return (
      <div className="flex min-h-[60vh] items-center justify-center p-8">
        <div className="max-w-md rounded-lg border border-line bg-surface-1 p-6">
          <h1 className="text-base font-semibold text-ink">Something broke on this page</h1>
          <p className="mt-2 text-sm leading-relaxed text-ink-muted">
            The rest of the application is unaffected — use the navigation to move
            elsewhere, or reload to try this page again.
          </p>

          <pre className="mt-4 overflow-x-auto rounded-md border border-line bg-surface-2 p-3
                          text-micro text-ink-faint">
            {error.message || String(error)}
          </pre>

          <div className="mt-5 flex gap-2">
            <button
              onClick={() => this.setState({ error: null })}
              className="rounded-md bg-brand px-3.5 py-2 text-sm font-medium text-surface-0
                         transition-colors hover:bg-brand-strong"
            >
              Try again
            </button>
            <button
              onClick={() => window.location.reload()}
              className="rounded-md border border-line px-3.5 py-2 text-sm font-medium
                         text-ink-muted transition-colors hover:border-line-strong hover:text-ink"
            >
              Reload
            </button>
          </div>
        </div>
      </div>
    );
  }
}
