import curses

stdscr = curses.initscr()                # 画面を初期化する
stdscr.addch(0, 0, "😃")                 # 1文字表示する
stdscr.addstr(0, 2, " < Hello, world!")  # 文字列を表示する
stdscr.refresh()                         # 画面の変更を反映する
stdscr.getch()                           # 入力を待ち受ける
curses.endwin()                          # cursesを終了する

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
    "simple.conf",
)
p = Population(c)

import curses
import itertools
import math
import time

from neat.nn import FeedForwardNetwork

def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        net = FeedForwardNetwork.create(genome, config)

        # ================ ドメインに依存する処理を実装する ==========
        genome.fitness = 0
        BLANK = " "
        GOAL = "🍪"
        AGENT = "😃"  # エージェントの生成
        GAME_CLEAR = "😍"
        GAME_OVER = "👽"

        goal = [10, 10]  # ゴール (この場所を探す)
        current = [30, 80]  # エージェントの開始位置

        stdscr = curses.initscr()  # 画面の初期化
        stdscr.addch(goal[0], goal[1], GOAL)  # ゴール

        for i in itertools.count():
            # 表示を更新
            stdscr.addstr(0, 0, f"GENOME: {genome.key} | life: {i} | current: {current} | fitness: {genome.fitness}                        ")
            if goal == current:  # ゴールに到達
                genome.fitness += 1000  # 報酬を追加

                stdscr.addstr(0, 0, f"GENOME: {genome.key} | life: {i} | current: {current} | fitness: {genome.fitness}                        ")
                stdscr.addch(current[0], current[1], GAME_CLEAR)
                stdscr.refresh()
                time.sleep(5)
                break

            if i > 100:  # 寿命に到達
                # ゴールと自分自身の距離を測る
                distance = math.sqrt(
                    (goal[0] - current[0]) ** 2 + (goal[1] - current[1]) ** 2
                )
                genome.fitness -= distance  # 報酬を追加

                # ゲームオーバー
                try:
                    stdscr.addstr(0, 0, f"GENOME: {genome.key} | life: {i} | current: {current} | fitness: {genome.fitness}                        ")
                    stdscr.addch(current[0], current[1], GAME_OVER)
                    stdscr.refresh()
                    time.sleep(0.3)
                    stdscr.addch(current[0], current[1], BLANK)
                except curses.error:  # 画面はみ出し（文字だけ表示）
                    stdscr.addstr(1, 1, f"DEAD")
                    stdscr.refresh()
                    time.sleep(0.3)
                    stdscr.addstr(1, 1, f"    ")
                    stdscr.refresh()
                break

            try:
                # エージェント描画
                stdscr.addch(current[0], current[1], AGENT)
                stdscr.refresh()
                time.sleep(0.01)
                stdscr.addch(current[0], current[1], BLANK)
            except curses.error:
                pass  # 画面はみ出し（無視する）

            # 移動
            input_data = [
                i,
                current[0],  # 現在位置
                current[1],  # 現在位置
            ]
            o_xy = net.activate(input_data)
            axis = 0 if o_xy[0] > o_xy[1] else 1
            amount = 1 if o_xy[axis] < 0.5 else -1

            stdscr.refresh()
            if (current[axis] + amount) > 1:
                current[axis] += amount
        # ================ ドメインに依存する処理ここまで ===========


winner = p.run(eval_genomes, n=100)  # 10世代
curses.endwin()  # ゲーム画面の終了
print(winner)