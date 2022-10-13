
from runners import ScenarioRunner

def main(args):
    
    scenario = 'assets/no_scenarios.json'
    route = 'assets/routes_phase0.xml'
    
    args.agent = 'autoagents/collector_agents/random_collector'
    args.agent_config = 'config.yaml'
    
    jobs = []
    for i in range(args.num_runners):
        port = (i+1) * 2000
        tm_port = port + 2
        runner = ScenarioRunner.remote(args, scenario, route, port=port, tm_port=tm_port)
        jobs.append(runner.run.remote())
    
    ray.wait(jobs, num_returns=args.num_runners)
    
if __name__ == '__main__':
    
    import argparse
    import ray
    ray.init(logging_level=40, local_mode=False, log_to_driver=False)

    parser = argparse.ArgumentParser()
    parser.add_argument('--num-runners', type=int, default=8)

    parser.add_argument('--host', default='localhost',
                        help='IP of the host server (default: localhost)')
    parser.add_argument('--trafficManagerSeed', default='0',
                        help='Seed used by the TrafficManager (default: 0)')
    parser.add_argument('--timeout', default="60.0",
                        help='Set the CARLA client timeout value in seconds')

    # agent-related options
    # parser.add_argument("-a", "--agent", type=str, help="Path to Agent's py file to evaluate", required=True)
    # parser.add_argument("--agent-config", type=str, help="Path to Agent's configuration file", default="")
    parser.add_argument('--repetitions',
                        type=int,
                        default=100,
                        help='Number of repetitions per route.')
    parser.add_argument("--track", type=str, default='MAP', help="Participation track: SENSORS, MAP")
    parser.add_argument('--resume', type=bool, default=False, help='Resume execution from last checkpoint?')
    parser.add_argument("--checkpoint", type=str,
                        default='./simulation_results.json',
                        help="Path to checkpoint used for saving statistics and resuming")
    
    args = parser.parse_args()
    
    main(args)
