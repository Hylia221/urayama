import numpy as np
# ランダムプレイヤー
class RandomPlayer:
    def __init__(self,kp):
        self.kp = kp
        self.random_count = 0

    def random_action_func(self):
        """
        出したカード5,宣言するカード3
        [0,0,0,0,0,1,0,0,0,0,1,0,0,0,0,0]
        """
        self.random_count += 1
        index = np.where(self.kp.p1_hand>0)
        if len(index[0]) > 0:
            act=np.random.choice(index[0])*8+np.random.choice(8)
            return act      
        else:
            self.kp.done=True
            self.kp.winner=-1
            return act
            
    def random_defence_action_func(self):
        self.random_count += 1
        return np.random.randint(0,2)##0:嘘、1:本当


        