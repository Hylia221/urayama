import numpy as np
import chainer
import chainerrl
import qfunction as qf
from functions import *
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
        self.p1_hand = np.array([self.p1_hand.count(x) for x in range(8)]).astype(np.float32)
        self.p2_hand = deck_tmp[int(len(deck_tmp)/2):]
        self.p2_hand = np.array([self.p2_hand.count(x) for x in range(8)]).astype(np.float32)
        self.p1_field=np.zeros(8)
        self.p2_field=np.zeros(8)
        self.winner = None
        self.miss = False
        self.done = False

    def step_defence(self,act):
        call = np.random.randint(64)
        true_card = np.int(call/8) #相手が出したカード
        declaration_card = call%8 #相手が宣言したカード
        if act==1:
            if true_card == declaration_card:#当てた
                kp.p2_field[true_card]+=1
                kp.p2_hand[np.random.choice(np.where(kp.p2_hand>0)[0])] -= 1
            else:#外れた
                kp.p1_field[true_card]+=1
        elif act==0:
            if true_card == declaration_card:#外れた
                kp.p1_field[true_card]+=1
            else:#当てた
                kp.p2_field[true_card]+=1
                kp.p2_hand[np.random.choice(np.where(kp.p2_hand>0)[0])]-=1

    def step_offence(self, act):
        call = np.random.randint(2)##0:嘘、1:本当
        true_card = np.int(act/8) #
        declaration_card = act%8 #
        self.p1_hand[true_card] -= 1 # 手札を減らす
        if call==1:
            if true_card == declaration_card:#当てられた
                kp.p1_field[true_card]+=1
                return -1
            else:
                kp.p2_field[true_card]+=1
                kp.p2_hand[np.random.choice(np.where(kp.p2_hand>0)[0])] -= 1
                return 1
        elif call==0:
            if true_card == declaration_card:
                kp.p2_field[true_card]+=1
                kp.p2_hand[np.random.choice(np.where(kp.p2_hand>0)[0])]-=1
                return 1
            else:#当てられた
                kp.p1_field[true_card]+=1
                return -1
        # return reward

    def get_env(self):
        a1 = self.p1_hand.tolist()
        a2 = self.p1_field.tolist()
        a3 = GRAY_CODE[int(sum(self.p2_hand))]
        a4 = self.p2_field.tolist()
        return np.array(a1+a2+a3+a4).astype(np.float32)
    
    def check_winner(self):
        # player1 win condtion
        if np.sum(self.p2_field>0)==8:
            self.winner=1
            self.done = True
        if np.sum(self.p2_field>4):
            self.winner=1
            self.done = True
        if np.sum(self.p2_hand)==0:
            self.winner=1
            self.done = True
        # player2 win condition
        if np.sum(self.p1_field>0)==8:
            self.winner=-1
            self.done = True
        if np.sum(self.p1_field>4):
            self.winner=-1
            self.done = True
        if np.sum(self.p1_hand)==0:
            self.winner=-1
            self.done = True

    def show(self):
        print("player1's hand:",self.p1_hand)
        print("player1's field:",self.p1_field)
        print("player2's hand:",self.p2_hand)
        print("player2's field:",self.p2_field)


# ゲームボードの準備
kp = Kakerlakenpoker()
p1_rndact = players.RandomPlayer(kp)
# 環境と行動の次元数
obs_size =29
n_actions = 64
#学習ゲーム回数
n_episodes = 3000
#カウンタの宣言
win = 0
miss = 0
# Q-functionとオプティマイザーのセットアップ
q_func = qf.QFunction(obs_size, n_actions)
# q_func.to_gpu(0)
optimizer = chainer.optimizers.Adam(eps=1e-2)
optimizer.setup(q_func)
# 報酬の割引率
gamma = 0.95
# Epsilon-greedyを使ってたまに冒険。50000ステップでend_epsilonとなる
p1_explorer = chainerrl.explorers.LinearDecayEpsilonGreedy(
    start_epsilon=1.0, end_epsilon=0.3, decay_steps=50000, random_action_func=p1_rndact.random_action_func)
# Experience ReplayというDQNで用いる学習手法で使うバッファ
replay_buffer = chainerrl.replay_buffer.ReplayBuffer(capacity=10 ** 6)
# Agentの生成（replay_buffer等を共有する2つ）
agent_p1 = chainerrl.agents.DoubleDQN(
    q_func, optimizer, replay_buffer, gamma, p1_explorer,
    replay_start_size=500,
    target_update_interval=100)

for i in range(1,n_episodes+1):
    kp.reset()
    reward=0
    turn = 0
    while not kp.done:
        action = agent_p1.act_and_train(kp.get_env().copy(),reward)
        tmp_r = kp.step_offence(action)
        reward += tmp_r
        kp.check_winner()
        if kp.done is True:
            if kp.winner == 1:
                reward += 100
                win += 1
            elif kp.winner == -1:
                reward += -100
            else:
                reward += -100
            if kp.miss is True:
                miss +=1
            agent_p1.stop_episode_and_train(kp.get_env().copy(), reward, True)
        else:
            # print("***Turn",turn,"***")
            # print(kp.show())
            last_state = kp.get_env().copy()
            turn +=1
    # print(reward)
    if i % 100 == 0:
        print("***Episodes",i,"***")
        print("win:",win)
        print("rnd:",p1_rndact.random_count)
        win=0
        miss=0
        p1_rndact.random_count=0
agent_p1.save("model3000")
            