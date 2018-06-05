import numpy as np
import chainer
import chainerrl
import qfunction as qf
import random
import players
#kakerlaken poker game board
class Kakerlakenpoker():
    def reset(self):
        deck = []
        for i in range(8):
            for j in range(8):
                deck.append(i)
        deck_tmp = random.sample(deck,54) #trash 10 cards

        self.p1_hand = deck_tmp[:int(len(deck_tmp)/2)]
        self.p1_hand = np.array([self.p1_hand.count(x) for x in range(8)])
        self.p2_hand = deck_tmp[int(len(deck_tmp)/2):]
        self.p2_hand = np.array([self.p2_hand.count(x) for x in range(8)])
        self.p1_field=np.zeros(8,dtype=np.int)
        self.p2_field=np.zeros(8,dtype=np.int)
        self.winner = None
    
    def check_winner(self):
        # player1 win condtion
        if np.sum(self.p2_field>0)==8:
            self.winner=1
        if np.sum(self.p2_field>4):
            self.winner=1
        if np.sum(self.p2_hand)==0:
            self.winner=1
        # player2 win condition
        if np.sum(self.p1_field>0)==8:
            self.winner=-1
        if np.sum(self.p1_field>4):
            self.winner=-1
        if np.sum(self.p1_hand)==0:
            self.winner=-1
    
    def move(self):
        pass

    def show(self):
        print("player1's hand:",self.p1_hand)
        print("player1's field:",self.p1_field)
        print("player2's hand:",self.p2_hand)
        print("player2's field:",self.p2_field)
# ゲームボードの準備
kp = Kakerlakenpoker()
p1_rndact = players.RandomPlayer(kp,1)
p2_rndact = players.RandomPlayer(kp,-1)
#学習ゲーム回数
n_episodes = 1
#カウンタの宣言
win = 0
lose = 0
# Q-functionとオプティマイザーのセットアップ
q_func = qf.QFunction(obs_size, n_actions)
optimizer = chainer.optimizers.Adam(eps=1e-2)
optimizer.setup(q_func)
# 報酬の割引率
gamma = 0.95
# Epsilon-greedyを使ってたまに冒険。50000ステップでend_epsilonとなる
p1_explorer = chainerrl.explorers.LinearDecayEpsilonGreedy(
    start_epsilon=1.0, end_epsilon=0.3, decay_steps=50000, random_action_func=ra.random_action_func)
p2_explorer = chainerrl.explorers.LinearDecayEpsilonGreedy(
    start_epsilon=1.0, end_epsilon=0.3, decay_steps=50000, random_action_func=ra.random_action_func)
# Experience ReplayというDQNで用いる学習手法で使うバッファ
replay_buffer = chainerrl.replay_buffer.ReplayBuffer(capacity=10 ** 6)
# Agentの生成（replay_buffer等を共有する2つ）
chainerrl.agents.DoubleDQN
agent_p1 = chainerrl.agents.DoubleDQN(
    q_func, optimizer, replay_buffer, gamma, p1_explorer,
    replay_start_size=500,
    target_update_interval=100)
agent_p2 = chainerrl.agents.DoubleDQN(
    q_func, optimizer, replay_buffer, gamma, p2_explorer,
    replay_start_size=500,
    target_update_interval=100)

for i in range(1,n_episodes+1):
    kp.reset()
    reward = 0
    turn = np.random.choice([0, 1])
    # while kp.winner is None:
    #     if kp.winner is None:
    #         if kp.winner == 1:
    #             reward = 1
    #             win += 1
    #         elif kp.winner == -1:
    #             reward = -1
    #             lose +=1


            