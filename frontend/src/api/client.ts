import axios from 'axios';
import type {
  Player, CurrentPlayer, LeagueStat, FeatureImportance,
  KPIs, AccuracyBand, PredictRequest, PredictResponse,
  MatchPrediction, TeamRating, Fixture, MarketInsights
} from '../types';

// Set VITE_API_URL at build time to point the deployed frontend at the
// deployed API. Vite inlines it, so it must exist when `npm run build` runs -
// on Vercel that means adding it as a project environment variable, not a
// runtime one. Falls back to the local API so `npm run dev` needs no setup.
const baseURL = import.meta.env.VITE_API_URL ?? 'http://localhost:8000';

// Render's free tier stops the service after 15 minutes idle and takes roughly
// 50 seconds to boot again. Axios defaults to no timeout but browsers give up
// well before that, and a failed first request makes a working app look dead.
// So: a generous timeout, one automatic retry, and an event the shell can use
// to explain the wait instead of showing an error.
const http = axios.create({ baseURL, timeout: 90_000 });

let inFlight = 0;
let slowTimer: ReturnType<typeof setTimeout> | null = null;

const signal = (name: 'api-slow' | 'api-ok') =>
  window.dispatchEvent(new CustomEvent(name));

http.interceptors.request.use(cfg => {
  if (inFlight++ === 0) {
    slowTimer = setTimeout(() => signal('api-slow'), 8000);
  }
  return cfg;
});

const settle = () => {
  if (--inFlight <= 0) {
    inFlight = 0;
    if (slowTimer) { clearTimeout(slowTimer); slowTimer = null; }
    signal('api-ok');
  }
};

http.interceptors.response.use(
  r => { settle(); return r; },
  async err => {
    settle();
    const cfg = err.config;
    // Retry once, and only for a cold start - a timeout or a dropped
    // connection. A 4xx/5xx means the server answered and retrying it would
    // just repeat the same error twice as slowly.
    if (cfg && !cfg._retried && !err.response) {
      cfg._retried = true;
      return http(cfg);
    }
    return Promise.reject(err);
  },
);

export const api = {
  getKPIs: () =>
    http.get<KPIs>('/api/kpis').then(r => r.data),

  getPlayers: (params?: {
    competition?: string; position?: string;
    season?: number; search?: string; limit?: number; offset?: number;
  }) =>
    http.get<{ data: Player[]; total: number }>('/api/players', { params }).then(r => r.data),

  getCurrent: (params?: { competition?: string; position?: string; limit?: number }) =>
    http.get<{ data: CurrentPlayer[] }>('/api/current', { params }).then(r => r.data),

  getLeagues: () =>
    http.get<{ data: LeagueStat[] }>('/api/leagues').then(r => r.data),

  getImportance: () =>
    http.get<{ data: FeatureImportance[] }>('/api/importance').then(r => r.data),

  getMarketInsights: () =>
    http.get<MarketInsights>('/api/market-insights').then(r => r.data),

  getAccuracyBands: (competition?: string) =>
    http.get<{ data: AccuracyBand[] }>('/api/accuracy-bands', {
      params: competition ? { competition } : {}
    }).then(r => r.data),

  predict: (body: PredictRequest) =>
    http.post<PredictResponse>('/api/predict', body).then(r => r.data),

  getLeaderboard: (params: { period: string; direction: string; limit: number }) =>
    http.get<{ data: unknown[]; period: string; source?: string; note?: string }>(
      '/api/leaderboard', { params }
    ).then(r => r.data),

  getPlayerHistory: (playerName: string) =>
    http.get<{ data: unknown[]; player: string }>(`/api/history/${encodeURIComponent(playerName)}`).then(r => r.data),

  // ── Match outcome prediction ──────────────────────────────────────────────

  getMatchCompetitions: () =>
    http.get<{ data: { code: string; name: string }[] }>('/api/match/competitions')
      .then(r => r.data),

  getMatchTeams: (competition?: string) =>
    http.get<{ data: string[] }>('/api/match/teams', {
      params: competition ? { competition } : {}
    }).then(r => r.data),

  getMatchRatings: (params?: { competition?: string; limit?: number }) =>
    http.get<{ data: TeamRating[] }>('/api/match/ratings', { params }).then(r => r.data),

  predictMatch: (body: {
    home_team: string; away_team: string; competition?: string;
    odds_home?: number; odds_draw?: number; odds_away?: number;
  }) =>
    http.post<MatchPrediction>('/api/match/predict', body).then(r => r.data),

  getMatchAccuracy: () =>
    http.get<{
      scored: number; pending: number;
      log_loss?: number; brier?: number; accuracy?: number;
      first?: string; last?: string;
      backtest_log_loss?: number; vs_backtest?: number;
      error?: string;
    }>('/api/match/accuracy').then(r => r.data),

  getFixtures: (days = 7) =>
    http.get<{ data: Fixture[]; count: number; predicted: number; error?: string }>(
      '/api/match/fixtures', { params: { days } }
    ).then(r => r.data),
};
