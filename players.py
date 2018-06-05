import numpy as np
# ランダムプレイヤー
class RandomPlayer:
    def __init__(self,kp,player):
        self.kp = kp
        self.random_count = 0
        self.player = player
    def random_action_func(self):
        self.random_count += 1
        if self.player == 1:
            index = np.where(self.kp.p1_hand>0)
            act=np.zeros(9),dtype=int)
            if len(index) > 0:
                act[np.random.choice(index)
                act[8]= np.random.choice([0, 1])
                return act#0:嘘,1:本当          
            else:
                raise("RandomChoiceError")
        elif self.player == -1:
            index = np.where(self.kp.p2_hand>0)
            act=np.zeros((2,8),dtype=int)
            if len(index) > 0:
                act[np.random.choice([0, 1])][np.random.choice(index)] = 1
                return act#0:嘘,1:本当
            else:
                raise("RandomChoiceError")

        