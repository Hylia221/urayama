import numpy as np
import chainer
import chainerrl
import qfunction as qf
from functions import *
import random
import players
import kakerlakenpoker

def train_offence():
    # ゲームボードの準備
    kp = kakerlakenpoker.Kakerlakenpoker()
    p1_rndact = players.RandomPlayer(kp,PLAYER1)
    p2_rndact = players.RandomPlayer(kp,PLAYER2)
    # 環境と行動の次元数
    obs_size =32
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
        start_epsilon=1.0, end_epsilon=0.3, decay_steps=50000, random_action_func=p1_rndact.random_offence_action_func)
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
        reward_avg=0
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
                    miss += 1
                agent_p1.stop_episode_and_train(kp.get_env().copy(), reward, True)
            else:
                # print("***Turn",turn,"***")
                # print(kp.show())
                last_state = kp.get_env().copy()
                turn +=1
        reward_avg += reward
        if i % 100 == 0:
            print("***Episodes",i,"***")
            print("win:",win)
            print("reward avg:",reward_avg/100)
            print("rnd:",p1_rndact.random_count)
            win=0
            reward_avg = 0
            miss=0
            p1_rndact.random_count=0
    agent_p1.save("offence_model3000")
    
def train_defence():
        # ゲームボードの準備
    kp = kakerlakenpoker.Kakerlakenpoker()
    p1_rndact = players.RandomPlayer(kp,PLAYER1)
    p2_rndact = players.RandomPlayer(kp,PLAYER2)
    # 環境と行動の次元数
    obs_size = 40
    n_actions = 2
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
        start_epsilon=1.0, end_epsilon=0.3, decay_steps=50000, random_action_func=p1_rndact.random_defence_action_func)
    # Experience ReplayというDQNで用いる学習手法で使うバッファ
    replay_buffer = chainerrl.replay_buffer.ReplayBuffer(capacity=10 ** 6)
    agent_p1 = chainerrl.agents.DoubleDQN(
        q_func, optimizer, replay_buffer, gamma, p1_explorer,
        replay_start_size=500,
        target_update_interval=100)

    for i in range(1,n_episodes+1):
        kp.reset()
        reward = 0
        reward_avg = 0
        turn = 0
        while not kp.done:
            off_act = p2_rndact.random_offence_action_func()
            off_act_vec = np.zeros(8,dtype = np.float32)
            off_act_vec[off_act%8]=1
            env = np.append(kp.get_env().copy(),off_act_vec)
            def_act = agent_p1.act_and_train(env.copy(),reward)
            reward += kp.step_defence(off_act,def_act)
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
                agent_p1.stop_episode_and_train(env.copy(), reward, True)
            else:
                # print("***Turn",turn,"***")
                # print(kp.show())
                last_state = kp.get_env().copy()
                turn +=1
        reward_avg += reward
        if i % 100 == 0:
            print("***Episodes",i,"***")
            print("win:",win)
            print("reward avg:",reward_avg/100)
            print("rnd:",p1_rndact.random_count)
            win=0
            reward_avg = 0
            miss=0
            p1_rndact.random_count=0
    agent_p1.save("defence_model3000")

# train_offence()
train_defence()