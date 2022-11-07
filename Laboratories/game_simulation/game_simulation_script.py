import numpy as np
import math
import random
import pandas as pd
import statistics
import argparse as ap


# return a random position in the map
def startingPosition():
    x = np.random.randint(0, area)
    y = np.random.randint(0, area)
    return x, y


# move +-1 in x and y dimension
def move(x, y):
    x_movement = np.random.randint(-1, 2)
    y_movement = np.random.randint(-1, 2)
    # print(x_movement, y_movement)
    new_x = x + x_movement
    new_y = y + y_movement

    if new_x < 0 or new_x >= area:  # new x position is not valid
        new_x = x
    if new_y < 0 or new_y >= area:  # new y position is not valid
        new_y = y

    return new_x, new_y


# find if two players are near
def isNear(coord1, coord2):
    distance = math.dist(coord1, coord2)
    if distance < 1.42:  # distance less than sqrt(2) the players are near
        return True
    else:
        return False


class Player:
    def __init__(self, id, x, y):
        self.position = (x, y)
        self.id = id
        self.killed = False
        self.killedOpponents = 0
        self.winner = False

    def setPosition(self, position):
        self.position = position

    def setKilled(self):
        self.killed = True

    def increasKilledOpponents(self):
        self.killedOpponents += 1

    def setIsWinner(self):
        self.winner = True

    def getId(self):
        return self.id

    def getPosition(self):
        return self.position

    def getKilledOpponents(self):
        return self.killedOpponents

    def isKilled(self):
        return self.killed

    def isWinner(self):
        return self.winner


parser = ap.ArgumentParser()
parser.add_argument('--nRuns', type=int, default=30)
parser.add_argument('--area', type=int, default=5)
parser.add_argument('--initialPlayers', type=int, default=4)
parser.add_argument('--movementSpeed', type=int, default=1)

args = parser.parse_args()

seed = 42
seeds = [np.random.randint(0, 100) for i in range(args.nRuns)]

# save statistics
df = pd.DataFrame(columns=['time', 'initial_players', 'area', 'mobility_speed', 'seed', 'avg_killed_opponents',
                           'winner_killed_opponents'])

for area in range(3, 22, 2):
    for inital_players in range(2, 15):
        for movement_speed in range(1, 4):
            for seed in seeds:
                print("area {} - initialPlayers {} - movement_speed {}".format(area, inital_players, movement_speed))

                print(seed)
                np.random.seed(seed)
                random.seed(seed)

                # area = args.area
                # inital_players = args.initialPlayers
                # remaining_players = inital_players
                # movement_speed = args.movementSpeed
                epoch = 1  # used to compute the time to win
                # max_turns = 20

                # print("PARAMETERS\n")
                # print("random seed: {}".format(seed))
                # print("area: {}x{} m^2".format(area, area))
                # print("initial players: {}".format(inital_players))
                # print("movement speed: {} m/s".format(movement_speed))

                players = []  # list of players
                remaining_players = []
                # populate players
                for id in range(0, inital_players):
                    x, y = startingPosition()
                    p = Player(id, x, y)
                    players.append(p)
                    remaining_players.append(p)

                epoch = 0
                # while (remaining_players > 1 and epoch < max_turns):
                while len(remaining_players) > 1:
                    # move each player
                    for p in remaining_players:
                        if p.isKilled() is True:  # skip killed player
                            continue

                        current_position = p.getPosition()
                        # compute new position
                        new_position = move(current_position[0], current_position[1])
                        p.setPosition(new_position)

                    # shooting
                    for p in remaining_players:
                        if p.isKilled() is True:  # skip killed player
                            continue

                        for _p in remaining_players:
                            if p.getId() == _p.getId():  # same player
                                continue
                            if _p.isKilled() is True or p.isKilled() is True:
                                continue

                            if isNear(p.getPosition(), _p.getPosition()):  # if players are near
                                if np.random.randint(0, 2) == 0:  # p wins
                                    p.increasKilledOpponents()
                                    _p.setKilled()
                                    remaining_players.remove(_p)
                                    # remaining_players += -1

                                else:  # _p wins
                                    _p.increasKilledOpponents()
                                    p.setKilled()
                                    remaining_players.remove(p)
                                    # remaining_players += -1

                    random.shuffle(remaining_players)  # shuffle list to change the order used to search for near players
                    epoch += 1

                if len(remaining_players) == 1:
                    for p in players:
                        if p.isKilled() is False:
                            p.setIsWinner()
                            # print(f'PLAYER [{p.getId()}] WINS!!')

                if len(remaining_players) == 0:
                    print("ERROR -> remaining_player = 0")

                print("\n")

                # compute average of killed opponents for each players
                killed_opponents = []
                winner_killed_opponents = -1
                for p in players:
                    killed_opponents.append(p.getKilledOpponents())
                    if p.isWinner() is True:
                        winner_killed_opponents = p.getKilledOpponents()  # save killed opponents of the winner

                l = [epoch / movement_speed, inital_players, area, movement_speed, seed, statistics.fmean(killed_opponents),
                     winner_killed_opponents]
                df.loc[len(df)] = l

# save dataframe
# df_name = "runs[{}]_area[{}]_initP[{}]_mvmS[{}].csv".format(args.nRuns, args.area, args.initialPlayers, args.movementSpeed)
df.to_csv("output_simulation.csv", index=False)
