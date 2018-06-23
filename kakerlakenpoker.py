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
        p1_hand = deck_tmp[:int(len(deck_tmp)/2)]
        p2_hand = deck_tmp[int(len(deck_tmp)/2):]
        self.hand=np.array([[p1_hand.count(x) for x in range(8)],[p2_hand.count(x) for x in range(8)]],dtype=np.float32)
        self.field=np.array([[0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0]])
        self.winner = None
        self.miss = False
        self.done = False

    # def step_defence(self,off_act,def_act):
    #     true_card = np.int(off_act/8) #相手が出したカード
    #     declaration_card = off_act%8 #相手が宣言したカード
    #     if def_act==1:
    #         if true_card == declaration_card:#当てた
    #             self.p2_field[true_card]+=1
    #             self.p2_hand[np.random.choice(np.where(self.p2_hand>0)[0])] -= 1
    #             return 1
    #         else:#外れた
    #             self.p1_field[true_card]+=1
    #             return -1
    #     elif def_act==0:
    #         if true_card == declaration_card:#外れた
    #             self.p1_field[true_card]+=1
    #             return -1
    #         else:#当てた
    #             self.p2_field[true_card]+=1
    #             self.p2_hand[np.random.choice(np.where(self.p2_hand>0)[0])]-=1
    #             return 1
    #     self.miss = True
    #     self.done = True
    #     return 0

    # def step_offence(self, act):
    #     call = np.random.randint(2)##0:嘘、1:本当
    #     true_card = np.int(act/8) #
    #     declaration_card = act%8 #
    #     self.p1_hand[true_card] -= 1 # 手札を減らす
    #     if call==1:
    #         if true_card == declaration_card:#当てられた
    #             self.p1_field[true_card]+=1
    #             return -1
    #         else:
    #             self.p2_field[true_card]+=1
    #             self.p2_hand[np.random.choice(np.where(self.p2_hand>0)[0])] -= 1
    #             return 1
    #     elif call==0:
    #         if true_card == declaration_card:
    #             self.p2_field[true_card]+=1
    #             self.p2_hand[np.random.choice(np.where(self.p2_hand>0)[0])]-=1
    #             return 1
    #         else:#当てられた
    #             self.p1_field[true_card]+=1
    #             return -1
        # return reward
    def step_and_reward(self,off_act,def_act,turn,):
        true_card = np.int(off_act/8) 
        declaration_card = off_act%8 
        self.hand[turn][true_card]-=1
        if def_act==1:
            if true_card == declaration_card:
                self.field[turn][true_card]+=1
                return -1 if turn == PLAYER1 else 1
            else:
                self.field[PLAYER2-turn][true_card]+=1
                return 1 if turn == PLAYER1 else -1
        elif def_act==0:
            if true_card == declaration_card:
                self.field[PLAYER2-turn][true_card]+=1
                return 1 if turn == PLAYER1 else -1
            else:
                self.field[turn][true_card]+=1
                return -1 if turn == PLAYER1 else 1

    def step(self,off_act,def_act,turn):
        is_turn_change = False
        true_card = np.int(off_act/8) 
        declaration_card = off_act%8 
        print("True:",true_card)
        self.hand[turn][true_card]-=1
        if def_act==1:
            if true_card == declaration_card:#Player2が当てた
                self.field[turn][true_card]+=1
                is_turn_change=False
            else:#外れた
                self.field[PLAYER2-turn][true_card]+=1
                is_turn_change=True
        elif def_act==0:
            if true_card == declaration_card:#外れた
                self.field[PLAYER2-turn][true_card]+=1
                is_turn_change=True
            else:#当てた
                self.field[turn][true_card]+=1
                is_turn_change=False
        return is_turn_change

    def get_env(self):
        a1 = self.hand[PLAYER1].tolist()
        a2 = self.field[PLAYER1].tolist()
        a3 = GRAY_CODE[int(sum(self.hand[PLAYER2]))]
        a4 = self.field[PLAYER2].tolist()
        return np.array(a1+a2+a3+a4,dtype=np.float32)
    
    def check_winner(self):
        # player1 win condtion
        if np.sum(self.field[PLAYER2]>0)==8:
            self.winner=1
            self.done = True
        if np.sum(self.field[PLAYER2]>4):
            self.winner=1
            self.done = True
        if np.sum(self.hand[PLAYER2])==0:
            self.winner=1
            self.done = True
        if np.sum(self.hand[PLAYER2]<0):
            self.winner=1
            self.miss = True
            self.done = True
        # player2 win condition
        if np.sum(self.field[PLAYER1]>0)==8:
            self.winner=-1
            self.done = True
        if np.sum(self.field[PLAYER1]>4):
            self.winner=-1
            self.done = True
        if np.sum(self.hand[PLAYER1])==0:
            self.winner=-1
            self.done = True
        if np.sum(self.hand[PLAYER1]<0):
            print("Illegal playing")
            self.winner=-1
            self.miss = True
            self.done = True

    def show(self):
        print("player1's hand:",self.hand[PLAYER1])
        print("player1's field:",self.field[PLAYER1])
        print("player2's hand:",self.hand[PLAYER2])
        print("player2's field:",self.field[PLAYER2])
    
    def show_vs_URAYAMA(self):
        print("URAYAMA hand:",self.hand[PLAYER1])
        print("URAYAMA field:",self.field[PLAYER1])
        print("PLAYER hand:",self.hand[PLAYER2])
        print("PLAYER field:",self.field[PLAYER2])


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
    turn = PLAYER1 #PLAYER1がurayama, PLAYER2がhuman
    turn_count=1
    while not kp.done:
        print("***Turn",str(turn_count),"***")
        kp.show_vs_URAYAMA()
        off_act = offence_act[turn](kp.get_env().copy())
        off_act_vec = np.zeros(8,dtype = np.float32)
        off_act_vec[off_act%8]=1
        if turn==PLAYER1:
            print("URAYAMA declare:"+str(off_act%8))
        else :
            print("Player declare:"+str(off_act%8))
        def_act = defence_act[PLAYER2-turn](np.append(kp.get_env().copy(),off_act_vec))
        ans = "True" if def_act==1 else "Lie"
        if turn==PLAYER1:
            print("Player answer:"+ans) 
        else :
            print("URAYAMA answer:"+ans)
        is_turn_change = kp.step(off_act,def_act,turn)
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
        if is_turn_change : turn = PLAYER1 if turn == PLAYER2 else PLAYER2 #ターンの交換
        turn_count+=1
if __name__ == "__main__":
    main()