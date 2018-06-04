import numpy as np
import random
#kakerlaken poker game board
class Kakerlakenpoker():
    def reset(self):
        deck = []
        for i in range(8):
            for j in range(8):
                deck.append(i)
        deck_tmp = random.sample(deck,54) #trash 10 cards
        self.p1_hand=np.array(sorted(deck_tmp[:int(len(deck_tmp)/2)]))
        self.p2_hand=np.array(sorted(deck_tmp[int(len(deck_tmp)/2):]))
        self.p1_field=np.zeros(8)
        self.p2_field=np.zeros(8)
        self.winner = None
    
    def check_winner(self):
        if np.sum(self.p1_field>0)==8:
            self.winner=-1
        if np.sum(self.p1_field>4):
            self.winner=-1
        if np.sum(self.p2_field>0)==8:
            self.winner=1
        if np.sum(self.p2_field>4):
            self.winner=1

kp = Kakerlakenpoker()
kp.reset()
kp.p2_field=np.array([1,1,0,5,1,1,1,1])
kp.check_winner()
print(kp.winner)