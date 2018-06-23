import numpy as np
from functions import *
# ランダムプレイヤー
class RandomPlayer:
    def __init__(self,kp,player):
        self.kp = kp
        self.player = player
        self.random_count = 0

    def random_offence_action_func(self):
        self.random_count += 1
        index = np.where(self.kp.hand[PLAYER1]>0) if self.player == PLAYER1 else  np.where(self.kp.hand[PLAYER2]>0) 
        if len(index[0]) > 0:
            act=np.random.choice(index[0])*8+np.random.choice(8)
            return act      
        else:
            self.kp.done=True
            self.kp.miss=True
            self.kp.winner= -1 if self.player == PLAYER1 else 1 
            return -1
            
    def random_defence_action_func(self):
        self.random_count += 1
        # act = np.random.randint(2,dtype=np.int)##0:嘘、1:本当
        act=np.random.choice([0,1])
        return  act

#人間のプレーヤー
class HumanPlayer:
    def offence_act(self, env):
        valid = False
        while not valid:
            try:
                true_card = input("Please choose 0-7 to put out: ")
                declaration_card = input("Please choose 0-7 to declare:")
                true_card = int(true_card)
                declaration_card = int(declaration_card)
                if true_card >= 0 and true_card <= 7 and declaration_card >= 0 and declaration_card <= 7:
                    valid = True
                    return true_card*8+declaration_card
                else:
                    print("Invalid action")
            except Exception as e:
                print(true_card + " or" + declaration_card +  " is invalid")

    def defence_act(self,env):
        valid = False
        while not valid:
            try:
                ans = input("Please choose 0-1 to answer whether it is a lie(0) or true(1):")
                ans = int(ans)
                if ans == 0 or ans == 1 :
                    valid = True
                    return ans
                else:
                    print("Invalid action")
            except Exception as e:
                print( ans +  " is invalid")




        