import ray
from copy import deepcopy
from leaderboard.leaderboard_evaluator import LeaderboardEvaluator
from leaderboard.utils.statistics_manager import StatisticsManager

class ChallengeRunner():
    def __init__(self, args, scenario, route, port=1000, tm_port=1002, debug=False):
        args = deepcopy(args)

        # Inject args
        args.scenario_class = 'route_scenario'
        args.port = port
        args.trafficManagerPort = tm_port
        args.scenarios = scenario
        args.routes = route
        args.debug = debug
        args.record = ''

        self.runner = LeaderboardEvaluator(args, StatisticsManager())
        self.args = args

    def run(self):
        return self.runner.run(self.args)
