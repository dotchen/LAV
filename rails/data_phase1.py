from runners import ScenarioRunner

def main(args):

    towns = {i: f'Town{i+1:02d}' for i in range(7)}
    towns.update({7: 'Town10HD'})
    towns = {0: 'Town10HD'}

    # scenario = 'assets/all_towns_traffic_scenarios.json'
    scenario = 'assets/no_scenarios.json'
    route = 'assets/routes_all.xml'
    # route = 'assets/routes_training/route_10.xml'

    args.agent = 'autoagents/collector_agents/q_collector' # Use 'viz_collector' for collecting pretty images
    # args.agent_config = 'config.yaml'
    args.agent_config = 'config_leaderboard.yaml'

    # args.agent = 'autoagents/collector_agents/lidar_q_collector'
    # args.agent_config = 'config_lidar.yaml'

    jobs = []
    for i in range(args.num_runners):
        # scenario_class = 'train_scenario' # Use 'nocrash_train_scenario' to collect NoCrash training trajs
        scenario_class = 'nocrash_train_scenario'
        town = towns.get(i, 'Town03')
        port = (i+1) * args.port
        tm_port = port + 2
        checkpoint = f'results/{i:02d}_{args.checkpoint}'
        runner = ScenarioRunner.remote(args, scenario_class, scenario, route, checkpoint=checkpoint, town=town, port=port, tm_port=tm_port)
        jobs.append(runner.run.remote())
    
    ray.wait(jobs, num_returns=args.num_runners)


if __name__ == '__main__':

    import argparse
    import ray
    ray.init(logging_level=40, local_mode=False, log_to_driver=True)

    parser = argparse.ArgumentParser()

    parser.add_argument('--num-runners', type=int, default=1)

    parser.add_argument('--host', default='localhost',
                        help='IP of the host server (default: localhost)')
    parser.add_argument('--port', type=int, default=2000)
    parser.add_argument('--trafficManagerSeed', default='0',
                        help='Seed used by the TrafficManager (default: 0)')
    parser.add_argument('--timeout', default="600.0",
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
                        default='simulation_results.json',
                        help="Path to checkpoint used for saving statistics and resuming")
    
    args = parser.parse_args()
    
    main(args)
