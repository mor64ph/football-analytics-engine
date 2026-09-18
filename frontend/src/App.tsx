import { lazy, Suspense } from 'react';
import { BrowserRouter, Route, Routes, useLocation } from 'react-router-dom';
import ErrorBoundary from './components/ErrorBoundary';
import Layout from './components/Layout';
import { Skeleton } from './components/ui';

/**
 * Routes are code-split.
 *
 * recharts is most of the bundle and only three screens draw charts, but a
 * single chunk made every visitor download all of it before anything rendered.
 * Splitting per route means the first paint carries the shell and one page.
 */
const Overview        = lazy(() => import('./pages/Overview'));
const PlayerSearch    = lazy(() => import('./pages/PlayerSearch'));
const Simulator       = lazy(() => import('./pages/Simulator'));
const LeagueComparison = lazy(() => import('./pages/LeagueComparison'));
const Leaderboard     = lazy(() => import('./pages/Leaderboard'));
const MatchPredictor  = lazy(() => import('./pages/MatchPredictor'));
const Explainability  = lazy(() => import('./pages/Explainability'));

function PageFallback() {
  return (
    <div className="space-y-6">
      <Skeleton className="h-8 w-64" />
      <div className="grid grid-cols-1 gap-4 md:grid-cols-2 xl:grid-cols-4">
        {Array.from({ length: 4 }).map((_, i) => <Skeleton key={i} className="h-24 w-full" />)}
      </div>
      <Skeleton className="h-72 w-full" />
    </div>
  );
}

/**
 * Resets the error boundary when the route changes, so recovering is a matter
 * of navigating away rather than reloading the tab.
 */
function RoutedContent() {
  const location = useLocation();
  return (
    <ErrorBoundary key={location.pathname}>
      <Suspense fallback={<PageFallback />}>
        <Routes>
          <Route path="/"               element={<Overview />} />
          <Route path="/players"        element={<PlayerSearch />} />
          <Route path="/simulator"      element={<Simulator />} />
          <Route path="/leagues"        element={<LeagueComparison />} />
          <Route path="/leaderboard"    element={<Leaderboard />} />
          <Route path="/matches"        element={<MatchPredictor />} />
          <Route path="/explainability" element={<Explainability />} />
        </Routes>
      </Suspense>
    </ErrorBoundary>
  );
}

export default function App() {
  return (
    <BrowserRouter>
      <Layout>
        <RoutedContent />
      </Layout>
    </BrowserRouter>
  );
}
