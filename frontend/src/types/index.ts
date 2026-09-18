export interface Player {
  player_name: string;
  competition: string;
  season: number;
  position_group: string;
  age: number;
  goals: number;
  assists: number;
  minutes_played: number;
  international_caps: number;
  club_prestige_eur: number;
  market_value_eur: number;
  predicted_value_eur: number;
  difference_eur: number;
  pct_error: number;
  abs_pct_error: number;
  accuracy_band: string;
}

export interface CurrentPlayer {
  player_name: string;
  team_name: string;
  competition: string;
  season: number;
  position: string;
  goals: number;
  assists: number;
  played_matches: number;
  age: number;
  predicted_value_eur: number;
}

export interface LeagueStat {
  competition: string;
  avg_actual_value: number;
  avg_predicted_value: number;
  avg_mape: number;
  total_players: number;
  within_25_pct: number;
}

export interface FeatureImportance {
  feature: string;
  importance: number;
}

export interface KPIs {
  avg_mape: number;
  within_10_pct: number;
  within_25_pct: number;
  total_players: number;
  total_records: number;
  seasons_covered: number[];
  competitions: string[];
  positions: string[];
}

export interface AccuracyBand {
  band: string;
  count: number;
}

export interface PredictRequest {
  age: number;
  goals_p90: number;
  assists_p90: number;
  minutes_played: number;
  position: string;
  competition: string;
  international_caps: number;
  club_prestige_eur: number;
}

export interface PredictResponse {
  predicted_value_eur: number;
  predicted_value_m: number;
  formatted: string;
}

// ── Market insights ─────────────────────────────────────────────────────────

export interface ValuedPlayer {
  player_name: string;
  competition: string;
  position: string;
  age: number;
  market_value_eur: number;
  predicted_value_eur: number;
  // predicted minus market: positive = the market is underpricing him.
  gap_eur: number;
  gap_pct: number;
  slot?: string;
}

export interface MarketInsights {
  available: boolean;
  season: number;
  total_value_eur: number;
  total_players: number;
  peak_age: number | null;
  most_undervalued: ValuedPlayer | null;
  most_overvalued: ValuedPlayer | null;
  age_curve: { age: number; median_value_eur: number; p75_value_eur: number; n: number }[];
  undervalued_xi: ValuedPlayer[];
  leagues: {
    competition: string; n: number; p25: number; median: number;
    p75: number; max: number; total_eur: number;
  }[];
  mispricing_by_age: { band: string; median_gap_eur: number; n: number }[];
}

// ── Match outcome prediction ────────────────────────────────────────────────

export interface Outcome {
  home_win: number;
  draw: number;
  away_win: number;
}

export interface MatchPrediction {
  home_team: string;
  away_team: string;
  competition: string;
  known_teams: boolean;
  probabilities: Outcome;
  // Present only when odds were supplied. Supplying them does NOT change
  // `probabilities` — blending was measured and put zero weight on this
  // model, so the odds are used for comparison only.
  market: (Outcome & { overround: number; edge: Outcome }) | null;
  elo: { home: number; away: number; diff: number };
  expected_goals: { home: number; away: number; total: number };
  goals_markets: {
    over_1_5: number;
    over_2_5: number;
    over_3_5: number;
    btts: number;
    home_clean_sheet: number;
    away_clean_sheet: number;
  };
  top_scores: { score: string; probability: number }[];
  scoreline_grid: number[][];
}

export interface TeamRating {
  team: string;
  elo: number;
  competition: string;
  attack: number | null;
  defence: number | null;
}

export interface Fixture {
  utc_date: string;
  competition: string | null;
  home_team: string;
  away_team: string;
  status: string;
  predicted: boolean;
  prediction?: MatchPrediction;
}
