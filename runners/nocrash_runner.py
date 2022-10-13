import os
import csv
import ray
from copy import deepcopy
from leaderboard.nocrash_evaluator import NoCrashEvaluator

class NoCrashEvalRunner():
    def __init__(self, args, town, weather, port=1000, tm_port=1002, debug=False):
        args = deepcopy(args)

        # Inject args
        args.scenario_class = 'nocrash_eval_scenario'
        args.port = port
        args.trafficManagerPort = tm_port
        args.debug = debug
        args.record = ''
        
        args.town = town
        args.weather = weather

        self.runner = NoCrashEvaluator(args, StatisticsManager(args))
        self.args = args

    def run(self):
        return self.runner.run(self.args)


class StatisticsManager:
    
    headers = [
        'town',
        'traffic',
        'weather',
        'start',
        'target',
        'route_completion',
        'lights_ran',
        'duration',
    ]
    
    def __init__(self, args):
        
        self.finished_tasks = {
            'Town01': {},
            'Town02': {}
        }
        
        logdir = args.agent_config.replace('.yaml', '.csv')
        
        if args.resume and os.path.exists(logdir):
            self.load(logdir)
            self.csv_file = open(logdir, 'a')
            self.csv_writer = csv.DictWriter(self.csv_file, fieldnames=self.headers)
        else:
            self.csv_file = open(logdir, 'w')
            self.csv_writer = csv.DictWriter(self.csv_file, fieldnames=self.headers)
            self.csv_writer.writeheader()

    def load(self, logdir):
        with open(logdir, 'r') as file:
            log = csv.DictReader(file)
            for row in log:
                self.finished_tasks[row['town']][(
                    int(row['traffic']),
                    int(row['weather']),
                    int(row['start']),
                    int(row['target']),
                )] = [
                    float(row['route_completion']),
                    int(row['lights_ran']),
                    float(row['duration']),
                ]
    
    def log(self, town, traffic, weather, start, target, route_completion, lights_ran, duration):
        self.csv_writer.writerow({
            'town'            : town,
            'traffic'         : traffic,
            'weather'         : weather,
            'start'           : start,
            'target'          : target,
            'route_completion': route_completion,
            'lights_ran'      : lights_ran,
            'duration'        : duration,
        })

        self.csv_file.flush()
        
    def is_finished(self, town, route, weather, traffic):
        start, target = route
        key = (int(traffic), int(weather), int(start), int(target))
        return key in self.finished_tasks[town]
