import logging

from MCTS.Coach import Coach
from MCTS.Game import Game
from MCTS.NeuralNet import NNetWrapper as nn
from Utils.utils import *

log = logging.getLogger(__name__)


args = dotdict({
    'numIters': 1000,
    'numEps': 100,              # Number of complete self-play games to simulate during a new iteration.
    'tempThreshold': 15,        #
    'updateThreshold': 0.6,     # During arena playoff, new neural net will be accepted if threshold or more of games are won.
    'maxlenOfQueue': 200000,    # Number of game examples to train the neural networks.
    'numMCTSSims': 50,          # Number of games moves for MCTS to simulate.
    'arenaCompare': 40,         # Number of games to play during arena play to determine if new net will be accepted.
    'cpuct': 1,
    'max_step': 50,

    'checkpoint': './temp/',
    'load_model': False,
    'load_folder_file': ('/dev/models/8x100x50','best.pth.tar'),
    'numItersForTrainExamplesHistory': 20,

})


starting_game_state = {'game': {'id': '90abe2d8-001b-4c61-95f8-c4fcc42710d9', 'ruleset': {'name': 'standard', 'version': 'cli', 'settings': {'foodSpawnChance': 15, 'minimumFood': 1, 'hazardDamagePerTurn': 14, 'hazardMap': '', 'hazardMapAuthor': '', 'royale': {'shrinkEveryNTurns': 25}, 'squad': {'allowBodyCollisions': False, 'sharedElimination': False, 'sharedHealth': False, 'sharedLength': False}}}, 'map': 'standard', 'timeout': 500, 'source': ''}, 'turn': 0, 'board': {'height': 11, 'width': 11, 'snakes': [{'id': '641b7f22-6c33-4f97-a4f6-d00a4b2db243', 'name': 'Pipe Snake', 'latency': '0', 'health': 100, 'body': [{'x': 1, 'y': 5}, {'x': 1, 'y': 5}, {'x': 1, 'y': 5}], 'head': {'x': 1, 'y': 5}, 'length': 3, 'shout': '', 'squad': '', 'customizations': {'color': '#f57f18', 'head': 'scarf', 'tail': 'rbc-necktie'}}, {'id': 'b03e289c-fb3b-4e9e-87d5-448f2975ede0', 'name': 'Ball Python', 'latency': '0', 'health': 100, 'body': [{'x': 5, 'y': 9}, {'x': 5, 'y': 9}, {'x': 5, 'y': 9}], 'head': {'x': 5, 'y': 9}, 'length': 3, 'shout': '', 'squad': '', 'customizations': {'color': '#f57f18', 'head': 'scarf', 'tail': 'rbc-necktie'}}], 'food': [{'x': 0, 'y': 4}, {'x': 6, 'y': 10}, {'x': 5, 'y': 5}], 'hazards': []}, 'you': {'id': 'b03e289c-fb3b-4e9e-87d5-448f2975ede0', 'name': 'Ball Python', 'latency': '0', 'health': 100, 'body': [{'x': 5, 'y': 9}, {'x': 5, 'y': 9}, {'x': 5, 'y': 9}], 'head': {'x': 5, 'y': 9}, 'length': 3, 'shout': '', 'squad': '', 'customizations': {'color': '#f57f18', 'head': 'scarf', 'tail': 'rbc-necktie'}}}

def main():
    log.info('Loading %s...', Game.__name__)
    game = Game(starting_game_state)

    log.info('Loading %s...', nn.__name__)
    nnet = nn(game)

    if args.load_model:
        log.info('Loading checkpoint "%s/%s"...', args.load_folder_file[0], args.load_folder_file[1])
        nnet.load_checkpoint(args.load_folder_file[0], args.load_folder_file[1])
    else:
        log.warning('Not loading a checkpoint!')

    log.info('Loading the Coach...')
    c = Coach(game, nnet, args)

    if args.load_model:
        log.info("Loading 'trainExamples' from file...")
        c.loadTrainExamples()

    log.info('Starting the learning process 🎉')
    c.learn()


if __name__ == "__main__":
    main()
