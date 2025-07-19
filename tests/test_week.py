import sys
import os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__) + '/../'))

from simulation.week import Week
from simulation.game import Game
import pytest

class DummyGame:
    def __init__(self, home_team, away_team, winner=None, prob=0.7):
        self.home_team = home_team
        self.away_team = away_team
        self._winner = winner if winner else home_team
        self._prob = prob
    def simulate(self):
        return self._winner, self._prob

def test_week_simulate_results_structure():
    games = [DummyGame('A', 'B', winner='A', prob=0.8), DummyGame('C', 'D', winner='D', prob=0.6)]
    week = Week(games)
    results = week.simulate()
    assert len(results) == 2
    for r in results:
        assert 'winner' in r and 'prob' in r and 'home_team' in r and 'away_team' in r
        assert r['winner'] in [r['home_team'], r['away_team']]
        assert 0 <= r['prob'] <= 1
