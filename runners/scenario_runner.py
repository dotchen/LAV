import ray
from copy import deepcopy
from leaderboard.leaderboard_evaluator import LeaderboardEvaluator
from leaderboard.utils.statistics_manager import StatisticsManager

@ray.remote(num_cpus=1./8, num_gpus=1./4, max_restarts=100, max_task_retries=-1)
class ScenarioRunner():
    def __init__(self, args, scenario_class, scenario, route, checkpoint='simulation_results.json', town=None, port=1000, tm_port=1002, debug=False):
        args = deepcopy(args)

        # Inject args
        args.scenario_class = scenario_class
        args.town = town
        args.port = port
        args.trafficManagerPort = tm_port
        args.scenarios = scenario
        args.routes = route
        args.debug = debug
        args.checkpoint = checkpoint
        args.record = ''

        self.runner = LeaderboardEvaluator(args, StatisticsManager())
        self.args = args

    def run(self):
        return self.runner.run(self.args)
