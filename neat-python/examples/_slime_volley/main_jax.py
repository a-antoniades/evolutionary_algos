from neat.config import Config
from neat.genes import DefaultNodeGene
from neat.genome import DefaultGenome
from neat.population import Population
from neat.reproduction import DefaultReproduction
from neat.species import DefaultSpeciesSet
from neat.stagnation import DefaultStagnation

c = Config(
    DefaultGenome,
    DefaultReproduction,
    DefaultSpeciesSet,
    DefaultStagnation,
    "simple_.conf",
)
p = Population(c)

import curses
import itertools
import functools
import math
import time

from tqdm import tqdm
import os

# os.environ['JAX_PLATFORM_NAME'] = 'cpu'
# from neat.nn import FeedForwardNetwork
from neat.nn import JAXFeedForwardNetwork
import jax
import jax.numpy as jnp
from evojax.task.slimevolley import SlimeVolley

import argparse
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')



def parse_args():
    parser = argparse.ArgumentParser(description='EvoJAX SlimeVolley')
    parser.add_argument('--max_steps', type=int, default=3000, help='Max steps')
    parser.add_argument('--num_episodes', type=int, default=1, help='Number of episodes')
    parser.add_argument('--goal', type=float, default=0.3, help='Goal')
    parser.add_argument('--debug', action='store_true', help='Debug mode')
    parser.add_argument('--n_generations', type=int, default=100, help='Number of generations')
    args = parser.parse_args()
    return args
                        

# Now, modify the eval_genomes function to include logging
def eval_genomes(genomes, config, args):
    master_key = jax.random.PRNGKey(0)
    
    train_task = SlimeVolley(test=False, max_steps=args.max_steps)
    test_task = SlimeVolley(test=True, max_steps=args.max_steps)

    current = 0
    for genome_id, genome in tqdm(genomes, desc='Evaluating genomes'):
        net = JAXFeedForwardNetwork.create(genome, config)

        # Initialize fitness and other statistics
        genome.fitness = 0
        episode_lengths = []
        episode_scores = []

        # Split the master key for each genome
        genome_keys = jax.random.split(master_key, num=args.num_episodes + 1)
        master_key = genome_keys[0]  # Update the master key for the next genome
        episode_keys = genome_keys[1:]  # Use the rest for the episodes

        # Run multiple episodes
        for episode, subkey in zip(tqdm(range(args.num_episodes), desc='Episodes'), episode_keys):
            subkey = jax.random.split(subkey, num=1)[0][None, :]
            task_state = train_task.reset(subkey)
            episode_score = 0
            episode_length = 0

            for i in itertools.count():
                input_data = task_state.obs
                input_data = input_data.squeeze(0)  # Assuming input_data has shape [1, input_dim]
                action = net.activate(input_data)
                action = jnp.expand_dims(action, axis=0)  # Assuming action needs to be [1, action_dim]

                task_state, reward, done = train_task.step(task_state, action)
                episode_score += reward
                episode_length += 1

                if done:
                    logging.info(f'Genome {genome_id}: Episode {episode} finished with score {episode_score} and length {episode_length}')
                    break

            # Update fitness and other statistics
            genome.fitness += episode_score
            episode_lengths.append(episode_length)
            episode_scores.append(episode_score)

        # Post-process statistics
        genome.fitness /= args.num_episodes  # Average fitness over episodes
        logging.info(f'Genome {genome_id}: Average fitness {genome.fitness}')

        # log episode lengths and scores
        logging.info(f'Genome {genome_id}: Episode lengths {episode_lengths}')
        logging.info(f'Genome {genome_id}: Episode scores {episode_scores}')


if __name__ == '__main__':
    args = parse_args()
    eval_genomes_ = functools.partial(eval_genomes, args=args)
    if args.debug is True:
        logging.info('Running in debug mode')
        import cProfile
        import pstats
        # Set up profiling
        profiler = cProfile.Profile()
        profiler.enable()

        # Run the main function to be profiled
        winner = p.run(eval_genomes_, n=1)

        # Disable profiling and print out stats
        profiler.disable()
        stats = pstats.Stats(profiler).sort_stats('cumtime')
        stats.print_stats()

        # Optionally, save stats to a file
        filename = 'profile.prof'
        if os.path.exists(filename):
            counter = 1
            while os.path.exists(f'profile_{counter}.prof'):
                counter += 1
            filename = f'profile_{counter}.prof'
        stats.dump_stats(filename)
    else:
        winner = p.run(eval_genomes_, n=args.n_generations)
        logging.info(f'Winner: {winner}')





# def eval_genomes(genomes, config):
#     key = jax.random.PRNGKey(0)[None, :]
#     train_task = SlimeVolley(test=False, max_steps=max_steps)
#     test_task = SlimeVolley(test=True, max_steps=max_steps)

#     for genome_id, genome in genomes:
#         net = JAXFeedForwardNetwork.create(genome, config)

#         genome.fitness = 0
#         goal = 0.3  
#         current = 0
        
#         task_state = train_task.reset(key)
#         for i in itertools.count():
#             if goal == current:
#                 genome.fitness += 1000
#                 break

#             if i > 100:
#                 score = current
#                 genome.fitness = score 

#             input_data = task_state.obs
#             # squeeze first dim
#             input_data = input_data.squeeze(0)
#             action = net.activate(input_data)
#             print(f"action shape: {action.shape}")
#             action = jnp.expand_dims(action, axis=0)
#             task_state, reward, done = train_task.step(task_state, action)


# winner = p.run(eval_genomes, n=100)  # 10世代
# print(winner)