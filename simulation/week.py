from .game import Game

class Week(object):
    def __init__(self, games):
        self.games = games  # List of Game objects

    def simulate(self):
        results = []
        for game in self.games:
            winner, prob = game.simulate()
            results.append({'home_team': game.home_team, 'away_team': game.away_team, 'winner': winner, 'prob': prob})
        return results
