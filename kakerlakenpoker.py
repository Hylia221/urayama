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

    def step_defence(self,off_act,def_act):
        true_card = np.int(off_act/8) #相手が出したカード
        declaration_card = off_act%8 #相手が宣言したカード
        if def_act==1:
            if true_card == declaration_card:#当てた
                self.p2_field[true_card]+=1
                self.p2_hand[np.random.choice(np.where(self.p2_hand>0)[0])] -= 1
                return 1
            else:#外れた
                self.p1_field[true_card]+=1
                return -1
        elif def_act==0:
            if true_card == declaration_card:#外れた
                self.p1_field[true_card]+=1
                return -1
            else:#当てた
                self.p2_field[true_card]+=1
                self.p2_hand[np.random.choice(np.where(self.p2_hand>0)[0])]-=1
                return 1
        self.miss = True
        self.done = True
        return 0

    def step_offence(self, act):
        call = np.random.randint(2)##0:嘘、1:本当
        true_card = np.int(act/8) #
        declaration_card = act%8 #
        self.p1_hand[true_card] -= 1 # 手札を減らす
        if call==1:
            if true_card == declaration_card:#当てられた
                self.p1_field[true_card]+=1
                return -1
            else:
                self.p2_field[true_card]+=1
                self.p2_hand[np.random.choice(np.where(self.p2_hand>0)[0])] -= 1
                return 1
        elif call==0:
            if true_card == declaration_card:
                self.p2_field[true_card]+=1
                self.p2_hand[np.random.choice(np.where(self.p2_hand>0)[0])]-=1
                return 1
            else:#当てられた
                self.p1_field[true_card]+=1
                return -1
        # return reward
    def step(self,off_act,def_act,turn):
        true_card = np.int(off_act/8) 
        declaration_card = off_act%8 
        if turn == PLAYER1:
            if def_act==1:
                if true_card == declaration_card:#当てた
                    self.p2_field[true_card]+=1
                else:#外れた
                    self.p1_field[true_card]+=1
            elif def_act==0:
                if true_card == declaration_card:#外れた
                    self.p1_field[true_card]+=1
                else:#当てた
                    self.p2_field[true_card]+=1
        elif turn == PLAYER2:
            if def_act==1:
                if true_card == declaration_card:#当てた
                    self.p1_field[true_card]+=1
                else:#外れた
                    self.p2_field[true_card]+=1
            elif def_act==0:
                if true_card == declaration_card:#外れた
                    self.p2_field[true_card]+=1
                else:#当てた
                    self.p1_field[true_card]+=1

    def get_env(self):
        a1 = self.p1_hand.tolist()
        a2 = self.p1_field.tolist()
        a3 = GRAY_CODE[int(sum(self.p2_hand))]
        a4 = self.p2_field.tolist()
        return np.array(a1+a2+a3+a4,dtype=np.float32)
    
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

def main():
    kp = Kakerlakenpoker()
    kp.reset()
    human_player = players.HumanPlayer()
    p1_rndact = players.RandomPlayer(kp,PLAYER1)
    # Q-functionとオプティマイザーのセットアップ
    off_q_func = qf.QFunction(32, 64)
    # q_func.to_gpu(0)
    off_optimizer = chainer.optimizers.Adam(eps=1e-2)
    off_optimizer.setup(off_q_func)
    gamma = 0.95
    # Epsilon-greedyを使ってたまに冒険。50000ステップでend_epsilonとなる
    off_explorer = chainerrl.explorers.LinearDecayEpsilonGreedy(
        start_epsilon=1.0, end_epsilon=0.3, decay_steps=50000, random_action_func=p1_rndact.random_offence_action_func)
    # Experience ReplayというDQNで用いる学習手法で使うバッファ
    off_replay_buffer = chainerrl.replay_buffer.ReplayBuffer(capacity=10 ** 6)
    urayama_offence = chainerrl.agents.DoubleDQN(
    off_q_func, off_optimizer, off_replay_buffer, gamma, off_explorer,
    replay_start_size=500,
    target_update_interval=100)

    # Q-functionとオプティマイザーのセットアップ
    def_q_func = qf.QFunction(40, 2)
    # q_func.to_gpu(0)
    def_optimizer = chainer.optimizers.Adam(eps=1e-2)
    def_optimizer.setup(def_q_func)
    def_explorer = chainerrl.explorers.LinearDecayEpsilonGreedy(
        start_epsilon=1.0, end_epsilon=0.3, decay_steps=50000, random_action_func=p1_rndact.random_defence_action_func)
    # Experience ReplayというDQNで用いる学習手法で使うバッファ
    def_replay_buffer = chainerrl.replay_buffer.ReplayBuffer(capacity=10 ** 6)

    urayama_defence =  chainerrl.agents.DoubleDQN(
    def_q_func, def_optimizer, def_replay_buffer, gamma, def_explorer,
    replay_start_size=500,
    target_update_interval=100)
    # chainerrl.agent.load_npz_no_strict("offence_model3000",urayama_offence)
    # chainerrl.agent.load_npz_no_strict("defence_model3000",urayama_defence)
    urayama_offence.load("offence_model3000")
    urayama_defence.load("defence_model3000")

    offence_act = [urayama_offence.act,human_player.offence_act]
    defence_act = [urayama_defence.act,human_player.defence_act]
    turn = PLAYER1 #0がurayama, 1がhuman
    turn_count=1
    while not kp.done:
        print("***Turn",turn_count,"***")
        print(kp.show())
        off_act = offence_act[turn](kp.get_env().copy())
        print("player"+turn+" declare "+off_act%8)
        def_act = defence_act[turn](np.append(kp.get_env().copy(),off_act))
        kp.step(off_act,def_act,turn)
        kp.check_winner()
        if kp.done is True:
            if kp.winner == 1:
                print("URAYAMA win")
            elif kp.winner == -1:
                print("YOU win")
            else:
                print("Error")
            if kp.miss is True:
                print("MISS")
        turn = PLAYER1 if turn == PLAYER2 else PLAYER2 #ターンの交換
        turn_count+=1
# main()