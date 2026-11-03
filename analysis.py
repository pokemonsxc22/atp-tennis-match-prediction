# %%
"""ATP match prediction. Run this script, or use the companion notebook."""
from pathlib import Path
from collections import defaultdict, deque
import hashlib
import json
import heapq
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, brier_score_loss

ROOT = Path(__file__).parent if '__file__' in globals() else Path.cwd()
DATA = ROOT / 'data'
OUT = ROOT / 'results'
OUT.mkdir(exist_ok=True)
manifest = json.loads((DATA / 'manifest.json').read_text())
for item in manifest['files']:
    assert hashlib.sha256((DATA / item['file']).read_bytes()).hexdigest() == item['sha256']
raw = pd.concat([pd.read_csv(DATA / item['file']) for item in manifest['files']], ignore_index=True)
raw['date'] = pd.to_datetime(raw['tourney_date'].astype(str), format='%Y%m%d')
keys = ['tourney_id', 'match_num', 'winner_id', 'loser_id']
raw = raw.drop_duplicates(keys).sort_values(['date', 'tourney_id', 'match_num'])
# Exclude team events, incomplete matches, unknown ranks and surfaces.
levels = ['250', '500', '1000', 'G', 'F', 'A', 'M']
valid = raw['tourney_level'].astype(str).isin(levels)
valid &= raw['surface'].isin(['Hard', 'Clay', 'Grass'])
valid &= ~raw['score'].fillna('').str.contains('RET|W/O|DEF|ABN', case=False, regex=True)
valid &= raw[['winner_rank', 'loser_rank']].notna().all(axis=1)
valid &= (raw[['winner_rank', 'loser_rank']] > 0).all(axis=1)
matches = raw.loc[valid].copy()
print(f'Loaded {len(raw):,} unique rows; retained {len(matches):,} completed, ranked tour matches.')

# %%
# Tournament dates are start dates, not exact match dates. Delay ALL historical
# outcomes 21 days after tournament start, avoiding within/overlapping-event leakage.
history = defaultdict(lambda: deque(maxlen=10))
surface_history = defaultdict(lambda: [0, 0])
pending = []
rows = []
for n, match in enumerate(matches.itertuples()):
    while pending and pending[0][0] < match.date:
        _, _, winner, loser, surface = heapq.heappop(pending)
        history[winner].append(1)
        history[loser].append(0)
        surface_history[(winner, surface)][0] += 1
        surface_history[(winner, surface)][1] += 1
        surface_history[(loser, surface)][1] += 1
    # Stable random orientation prevents the winner column from becoming the label.
    key = f'{match.tourney_id}|{match.match_num}|{match.winner_id}|{match.loser_id}'
    winner_is_a = int(hashlib.sha256(key.encode()).hexdigest()[:8], 16) % 2 == 0
    a, b = (match.winner_id, match.loser_id) if winner_is_a else (match.loser_id, match.winner_id)
    rank_a, rank_b = (match.winner_rank, match.loser_rank) if winner_is_a else (match.loser_rank, match.winner_rank)
    name_a, name_b = (match.winner_name, match.loser_name) if winner_is_a else (match.loser_name, match.winner_name)
    def form(player):
        results = history[player]
        return (sum(results) + 2) / (len(results) + 4)
    def surface_rate(player):
        wins, total = surface_history[(player, match.surface)]
        return (wins + 5) / (total + 10)
    rank_signal = np.log(rank_b / rank_a)
    rows.append({'match_id': key, 'date': match.date, 'surface': match.surface,
                 'player_a': name_a, 'player_b': name_b, 'target': int(winner_is_a),
                 'rank_signal': rank_signal, 'form_diff': form(a) - form(b),
                 'surface_diff': surface_rate(a) - surface_rate(b),
                 'rank_clay': rank_signal * (match.surface == 'Clay'),
                 'rank_grass': rank_signal * (match.surface == 'Grass')})
    heapq.heappush(pending, (match.date + pd.Timedelta(days=21), n, match.winner_id, match.loser_id, match.surface))
features = pd.DataFrame(rows)
columns = ['rank_signal', 'form_diff', 'surface_diff', 'rank_clay', 'rank_grass']
train = features[features.date.between('2016-01-01', '2022-12-31')]
validation = features[features.date.dt.year == 2023]
test = features[features.date.dt.year.isin([2024, 2025])]
assert train.date.max() < validation.date.min() < test.date.min()
assert not train.match_id.isin(test.match_id).any()
assert features.match_id.is_unique
print({'training': len(train), 'validation': len(validation), 'test': len(test)})

# %%
def fit_model(frame, c):
    # Mirroring TRAINING examples makes predictions invariant to player order.
    x = frame[columns].to_numpy()
    y = frame.target.to_numpy()
    model = make_pipeline(StandardScaler(with_mean=False), LogisticRegression(C=c, fit_intercept=False, max_iter=2000))
    model.fit(np.concatenate([x, -x]), np.concatenate([y, 1-y]))
    return model

scores = []
for c in [0.01, 0.1, 1.0]:
    candidate = fit_model(train, c)
    scores.append((log_loss(validation.target, candidate.predict_proba(validation[columns].to_numpy())[:, 1]), c))
best_loss, best_c = min(scores)
model = fit_model(pd.concat([train, validation]), best_c)
probability = model.predict_proba(test[columns].to_numpy())[:, 1]
assert np.allclose(probability + model.predict_proba(-test[columns].to_numpy())[:, 1], 1)
baseline = (test.rank_signal > 0).astype(float).to_numpy(copy=True)
baseline[test.rank_signal.to_numpy() == 0] = 0.5
records = []
for name, p in [('Logistic regression', probability), ('Higher-ranked player', baseline)]:
    records.append({'model': name, 'accuracy': accuracy_score(test.target, p >= 0.5),
                    'log_loss': log_loss(test.target, np.clip(p, 1e-6, 1-1e-6)),
                    'brier_score': brier_score_loss(test.target, p)})
metrics = pd.DataFrame(records)
print(metrics.to_string(index=False))
print('Baseline probabilities are hard 0/1 decisions; its log loss is not a calibrated forecast comparison.')
predictions = test[['match_id', 'date', 'surface', 'player_a', 'player_b', 'target']].copy()
predictions['p_a_wins'] = probability
predictions['predicted_winner'] = np.where(probability >= 0.5, test.player_a, test.player_b)
predictions.to_csv(OUT / 'test_predictions.csv', index=False)
metrics.to_csv(OUT / 'metrics.csv', index=False)
summary = {'train_matches': len(train), 'validation_matches': len(validation), 'test_matches': len(test),
           'best_c': best_c, 'data_commit': manifest['commit'], 'history_lag_days': 21,
           'metrics': records}
(OUT / 'summary.json').write_text(json.dumps(summary, indent=2))
print('\nHistorical examples (not live forecasts):')
print(predictions[['player_a', 'player_b', 'p_a_wins', 'predicted_winner']].head(6).to_string(index=False))

# %%
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].bar(metrics.model, metrics.accuracy * 100, color=['#2563eb', '#94a3b8'])
axes[0].set(ylabel='Accuracy (%)', ylim=(0, 100), title=f'2024–2025 backtest: {len(test):,} matches')
for i, value in enumerate(metrics.accuracy):
    axes[0].text(i, value * 100 + 1, f'{value:.1%}', ha='center')
predictions['bin'] = pd.cut(predictions.p_a_wins, np.linspace(0, 1, 11), include_lowest=True)
calibration = predictions.groupby('bin', observed=True).agg(predicted=('p_a_wins', 'mean'), actual=('target', 'mean'), count=('target', 'size'))
axes[1].plot(calibration.predicted, calibration.actual, 'o-', label='Model')
axes[1].plot([0, 1], [0, 1], '--', color='gray', label='Ideal')
axes[1].set(xlabel='Predicted probability', ylabel='Observed win rate', title='Probability calibration', xlim=(0, 1), ylim=(0, 1))
axes[1].legend()
fig.tight_layout()
fig.savefig(OUT / 'backtest.png', dpi=160)
if 'get_ipython' in globals():
    plt.show()
else:
    plt.close(fig)
