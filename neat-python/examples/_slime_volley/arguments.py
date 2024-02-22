import argparse



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--pop-size', type=int, default=128, help='ES population size.')
    parser.add_argument(
        '--hidden-size', type=int, default=20, help='Policy hidden size.')
    parser.add_argument(
        '--num-tests', type=int, default=100, help='Number of test rollouts.')
    parser.add_argument(
        '--n-repeats', type=int, default=16, help='Training repetitions.')
    parser.add_argument(
        '--max-iter', type=int, default=500, help='Max training iterations.')
    parser.add_argument(
        '--test-interval', type=int, default=50, help='Test interval.')
    parser.add_argument(
        '--log-interval', type=int, default=10, help='Logging interval.')
    parser.add_argument(
        '--seed', type=int, default=123, help='Random seed for training.')
    parser.add_argument(
        '--init-std', type=float, default=0.5, help='Initial std.')
    parser.add_argument(
        '--gpu-id', type=str, default="1", help='GPU(s) to use.')
    parser.add_argument(
        '--debug', action='store_true', help='Debug mode.')
    config, _ = parser.parse_known_args()
    return config


# def parse_neat_args():
#     parser = argparse.ArgumentParser(description="NEAT Configuration Parser")

#     # NEAT General Configuration
#     parser.add_argument("--fitness_criterion", type=str, default="max", help="Fitness criterion to use")
#     parser.add_argument("--fitness_threshold", type=float, default=100, help="Fitness threshold for termination")
#     parser.add_argument("--pop_size", type=int, default=10, help="Population size")
#     parser.add_argument("--reset_on_extinction", action="store_true", help="Whether to reset on extinction")
#     parser.add_argument("--no_fitness_termination", action="store_true", help="Disable fitness termination")

#     # DefaultGenome Configuration
#     parser.add_argument("--num_inputs", type=int, default=3, help="Number of input nodes")
#     parser.add_argument("--num_hidden", type=int, default=1, help="Number of hidden nodes")
#     parser.add_argument("--num_outputs", type=int, default=2, help="Number of output nodes")
#     parser.add_argument("--feed_forward", action="store_true", help="Use feed forward network")
#     parser.add_argument("--initial_connection", type=str, default="partial_direct", help="Initial connection strategy")
#     parser.add_argument("--initial_connection_prob", type=float, default=0.5, help="Probability for 'partial_direct' initial connection")

#     # Node Activation Options
#     parser.add_argument("--activation_default", type=str, default="sigmoid", help="Default activation function")
#     parser.add_argument("--activation_mutate_rate", type=float, default=0.0, help="Mutation rate for activation functions")

#     # Node Aggregation Options
#     parser.add_argument("--aggregation_default", type=str, default="sum", help="Default aggregation function")
#     parser.add_argument("--aggregation_mutate_rate", type=float, default=0.0, help="Mutation rate for aggregation functions")

#     # Connection Add/Remove Rates
#     parser.add_argument("--conn_add_prob", type=float, default=0.5, help="Probability of adding a new connection")
#     parser.add_argument("--conn_delete_prob", type=float, default=0.5, help="Probability of deleting a connection")

#     # Node Add/Remove Rates
#     parser.add_argument("--node_add_prob", type=float, default=0.2, help="Probability of adding a new node")
#     parser.add_argument("--node_delete_prob", type=float, default=0.2, help="Probability of deleting a node")

#     # Connection Enable Options
#     parser.add_argument("--enabled_default", action="store_true", help="Default state for new connections")
#     parser.add_argument("--enabled_mutate_rate", type=float, default=0.01, help="Mutation rate for enabling/disabling connections")

#     # Node Bias Options
#     parser.add_argument("--bias_init_mean", type=float, default=0.0, help="Initial mean for node biases")
#     parser.add_argument("--bias_init_stdev", type=float, default=1.0, help="Initial standard deviation for node biases")
#     parser.add_argument("--bias_max_value", type=float, default=30.0, help="Maximum value for node bias")
#     parser.add_argument("--bias_min_value", type=float, default=-30.0, help="Minimum value for node bias")
#     parser.add_argument("--bias_mutate_power", type=float, default=0.5, help="Mutation power for node bias")
#     parser.add_argument("--bias_mutate_rate", type=float, default=0.7, help="Mutation rate for node bias")
#     parser.add_argument("--bias_replace_rate", type=float, default=0.1, help="Replace rate for node bias")

#     # Node Response Options (similar structure for arguments as for node bias options)
#     parser.add_argument("--response_init_mean", type=float, default=1.0, help="Initial mean for node response")
#     parser.add_argument("--response_init_stdev", type=float, default=0.0, help="Initial standard deviation for node response")
#     parser.add_argument("--response_max_value", type=float, default=30.0, help="Maximum value for node response")
#     parser.add_argument("--response_min_value", type=float, default=-30.0, help="Minimum value for node response")
#     parser.add_argument("--response_mutate_power", type=float, default=0.0, help="Mutation power for node response")
#     parser.add_argument("--response_mutate_rate", type=float, default=0.0, help="Mutation rate for node response")
#     parser.add_argument("--response_replace_rate", type=float, default=0.0, help="Replace rate for node response")

#     # Connection Weight Options
#     parser.add_argument("--weight_init_mean", type=float, default=0.0, help="Initial mean for connection weights")
#     parser.add_argument("--weight_init_stdev", type=float, default=1.0, help="Initial standard deviation for connection weights")
#     parser.add_argument("--weight_max_value", type=float, default=30, help="Maximum value for connection weight")
#     parser.add_argument("--weight_min_value", type=float, default=-30, help="Minimum value for connection weight")
#     parser.add_argument("--weight_mutate_power", type=float, default=0.5, help="Mutation power for connection weight")
#     parser.add_argument("--weight_mutate_rate", type=float, default=0.8, help="Mutation rate for connection weight")
#     parser.add_argument("--weight_replace_rate", type=float, default=0.1, help="Replace rate for connection weight")

#     # Genome Compatibility Options
#     parser.add_argument("--compatibility_disjoint_coefficient", type=float, default=1.0, help="Coefficient for disjoint genes in compatibility calculation")
#     parser.add_argument("--compatibility_weight_coefficient", type=float, default=0.5, help="Coefficient for weight differences in compatibility calculation")

#     # DefaultSpeciesSet Configuration
#     parser.add_argument("--compatibility_threshold", type=float, default=3.3, help="Compatibility threshold for speciation")

#     # DefaultStagnation Configuration
#     parser.add_argument("--max_stagnation", type=int, default=100, help="Number of generations without improvement before a species is considered stagnant")
#     parser.add_argument("--species_elitism", type=int, default=1, help="Number of top species to be preserved over generations")

#     # DefaultReproduction Configuration
#     parser.add_argument("--elitism", type=int, default=2, help="Number of best individuals to carry over to the next generation")
#     parser.add_argument("--survival_threshold", type=float, default=0.1, help="Proportion of individuals surviving to the next generation")
#     parser.add_argument("--min_species_size", type=int, default=2, help="Minimum species size")

#     args = parser.parse_args()
#     return args