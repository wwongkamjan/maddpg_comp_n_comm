import argparse
import numpy as np
import tensorflow as tf
import time
import pickle
from tensorflow.contrib import rnn
import maddpg.common.tf_util as U
from maddpg.trainer.maddpg import MADDPGAgentTrainer
from maddpg.trainer.cmaddpg import CMADDPGAgentTrainer
import tensorflow.contrib.layers as layers

def parse_args():
    parser = argparse.ArgumentParser("Reinforcement Learning experiments for multiagent environments")
    # Environment
    parser.add_argument("--scenario", type=str, default="simple", help="name of the scenario script")
    parser.add_argument("--max-episode-len", type=int, default=25, help="maximum episode length")
    parser.add_argument("--num-episodes", type=int, default=60000, help="number of episodes")
    parser.add_argument("--num-adversaries", type=int, default=0, help="number of adversaries")
    parser.add_argument("--good-policy", type=str, default="maddpg", help="policy for good agents")
    parser.add_argument("--adv-policy", type=str, default="maddpg", help="policy of adversaries")
    # Core training parameters
    parser.add_argument("--lr", type=float, default=1e-2, help="learning rate for Adam optimizer")
    parser.add_argument("--gamma", type=float, default=0.95, help="discount factor")
    parser.add_argument("--batch-size", type=int, default=1024, help="number of episodes to optimize at the same time")
    parser.add_argument("--num-units", type=int, default=64, help="number of units in the mlp")
    # Checkpointing
    parser.add_argument("--exp-name", type=str, default=None, help="name of the experiment")
    parser.add_argument("--save-dir", type=str, default="./policy/", help="directory in which training state and model should be saved")
    parser.add_argument("--save-rate", type=int, default=1000, help="save model once every time this many episodes are completed")
    parser.add_argument("--load-dir", type=str, default="", help="directory in which training state and model are loaded")
    # Evaluation
    parser.add_argument("--restore", action="store_true", default=False)
    parser.add_argument("--display", action="store_true", default=False)
    parser.add_argument("--benchmark", action="store_true", default=False)
    parser.add_argument("--benchmark-iters", type=int, default=100000, help="number of iterations run for benchmarking")
    parser.add_argument("--benchmark-dir", type=str, default="./benchmark_files/", help="directory where benchmark data is saved")
    parser.add_argument("--plots-dir", type=str, default="./learning_curves/", help="directory where plot data is saved")
    return parser.parse_args()
    
# message encoding network 
def m_model(input, num_outputs, num_other, scope, num_layers=2, reuse=False, num_units=128, rnn_cell=None):
    hidden_size = num_units
    timestep_size = num_other
    with tf.variable_scope(scope, reuse=reuse):
        mlstm_cell=[]
        for _ in range(num_layers):    
            lstm_cell = tf.nn.rnn_cell.LSTMCell(num_units=hidden_size, forget_bias=1.0, state_is_tuple=True)
            mlstm_cell.append(lstm_cell)
        mlstm_cell = tf.nn.rnn_cell.MultiRNNCell(mlstm_cell, state_is_tuple=True)
        outputs, state = tf.nn.dynamic_rnn(cell=mlstm_cell, inputs=input, dtype=tf.float32)
        out = outputs[:, -1, :]
        out = layers.fully_connected(out, num_outputs=num_outputs, activation_fn=None)
        return out

def mlp_model(input, num_outputs, scope, type='fit', num_layer=3, reuse=False, num_units=128, rnn_cell=None):
    with tf.variable_scope(scope, reuse=reuse):
        out = input
        for i in range(num_layer-1):
            out = layers.fully_connected(out, num_outputs=num_units, activation_fn=tf.nn.relu)
        out = layers.fully_connected(out, num_outputs=num_outputs, activation_fn=None)
        return out

def make_env(arglist):
    from multiagent.environment import MultiAgentEnv
    import multiagent.scenarios as scenarios
    scenario_name = arglist.scenario
    benchmark = arglist.benchmark
    # load scenario from script
    scenario = scenarios.load(scenario_name + ".py").Scenario()
    # create world
    world = scenario.make_world()
    # create multiagent environment
    if benchmark:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, scenario.benchmark_data)
    else:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation)
    return env

def get_trainers(arglist, env, num_adversaries, obs_shape_n, message_shape_n,target_loc_space_n):
    trainers = []
    model = [mlp_model, m_model]
    trainer = CMADDPGAgentTrainer
    for i in range(num_adversaries):
        trainers.append(trainer(
            "agent_%d" % i, model, obs_shape_n, message_shape_n, target_loc_space_n, env.action_space, i, env.n_agents_obs, arglist,
            local_q_func=(arglist.adv_policy=='ddpg')))
    for i in range(num_adversaries, env.n_agents):
        trainers.append(trainer(
            "agent_%d" % i, model, obs_shape_n, message_shape_n, target_loc_space_n, env.action_space, i, env.n_agents_obs, arglist,
            local_q_func=(arglist.good_policy=='ddpg')))
    return trainers

def get_message(obs_n, target_pos_idx_n, num_agents_obs):
    num_agents = len(obs_n)
    obs_dim = obs_n[0].shape[-1]
    # if there exists no communication with other agents, message matrix will be zeros
    # messgae [zeros,zeros,o3,o4,zeros] for agent 1 when it communicates with 3,4
    message_n = [ np.zeros((num_agents_obs, obs_dim), dtype=np.float32) for _ in range(num_agents)]
    real_pos_n = []
    for j in range(num_agents):
        for jj in range(len(target_pos_idx_n[j])):
            message_n[j][jj,:] = obs_n[target_pos_idx_n[j][jj]]
    return message_n

def get_comm_pairs(obs_n, num_agents_obs, num_others):
    num_agents = len(obs_n)
    obs_dim = obs_n[0].shape[-1]
    target_loc_n = []
    target_idx_n = []
    target_idx = None
    real_loc_n = []
    # get the positions of each agent, the first two elements are the vel of each agent
    for i in range(num_agents):
        real_loc_n.append(obs_n[i][2:4])
    for i in range(num_agents):
        # remove the real_position and vel of agent, keep the relative position
        obs_tmp = obs_n[i][4:].copy()
        obs_tmp[0::2] = obs_tmp[0::2]+real_loc_n[i][0]
        obs_tmp[1::2] = obs_tmp[1::2]+real_loc_n[i][1]
        target_loc_all = []
        target_idx_all = []
        for j in range(num_agents_obs):
            target_loc = obs_tmp[int((num_others+j)*2): int((num_others+j)*2+2)]
            for ii in range(len(real_loc_n)):
                if (abs(real_loc_n[ii][0]-target_loc[0])<1e-5) and (abs(real_loc_n[ii][1]-target_loc[1])<1e-5):
                    target_idx = ii
            #tar_pos_all.append(obs_n[i][int((num_landmark+j)*2+4): int((num_landmark+j)*2+2+4)])
            target_loc_all.append(real_loc_n[i]-target_loc)
            target_idx_all.append(target_idx)
        target_loc_n.append(target_loc_all)
        target_idx_n.append(target_idx_all)
    return target_loc_n, target_idx_n

def train(arglist):
    with U.make_session():
        # Create environment
        env = make_env(arglist)
        obs_n = env.reset()
        num_others = env.n_landmarks_obs if arglist.scenario == 'cn' else env.n_preys_obs
        other_loc_n, other_idx_n = get_comm_pairs(obs_n, env.n_agents_obs, num_others)
        # Create agent trainers

        obs_shape_n = [env.observation_space[0].shape for i in range(env.n_agents)]
        message_shape_n = [ (env.n_agents_obs,)+env.observation_space[0].shape for _ in range(env.n_agents)]
        target_loc_space_n = [(len(other_loc_n[0][0]),) for _ in range(env.n_agents)]
        num_adversaries = min(env.n_agents, arglist.num_adversaries)
        trainers = get_trainers(arglist, env, num_adversaries, obs_shape_n, message_shape_n, target_loc_space_n)
        


        # Load previous results, if necessary
        if arglist.load_dir == "":
            arglist.load_dir = arglist.save_dir
        if arglist.display or arglist.restore or arglist.benchmark:
            print('Loading previous state...')
            for i in range(env.n_agents):
                trainers[i].initial_q_model()
                trainers[i].initial_p_model()
            U.initialize()
            U.load_state(arglist.load_dir)
        else:
            print('training MADDPG...')
            for i in range(env.n_agents):
                trainers[i].initial_q_model()
                trainers[i].initial_p_model()
            U.initialize()
        episode_rewards = [0.0]  # sum of rewards for all agents
        comm_freq = [0.0]
        agent_rewards = [[0.0] for _ in range(env.n_agents)]  # individual agent reward
        final_ep_rewards = []  # sum of rewards for training curve
        final_ep_ag_rewards = []  # agent rewards for training curve
        agent_info = [[[]]]  # placeholder for benchmarking info
        saver = tf.train.Saver()
        episode_step = 0
        training_step = 0
        t_start = time.time()
        max_mean_epi_reward = -100000
        num_comm = 0
        print('Starting iterations...')
        while True:
            # get messages
            action_n = []
            target_idx_n = [] 
            for i, agent in enumerate(trainers):
                other_loc = other_loc_n[i]
                other_idx = other_idx_n[i]
                target_idx = []
                for j in range(len(other_loc)):
                    if agent.target_comm(obs_n[i], np.array(other_loc[j])):
                        target_idx.append(other_idx[j])
                num_comm += len(target_idx)
                target_idx_n.append(target_idx)
          
            message_n = get_message(obs_n, target_idx_n, env.n_agents_obs)

            # get action
            action_n = [agent.action(obs, message) for agent, obs, message in zip(trainers,obs_n, message_n)]
            # environment step
            new_obs_n, rew_n, done_n, info_n = env.step(action_n)
            new_other_loc_n, new_other_idx_n = get_comm_pairs(new_obs_n, env.n_agents_obs, num_others)
            episode_step += 1
            done = all(done_n)
            terminal = (episode_step >= arglist.max_episode_len)
            # collect experience
            for i, agent in enumerate(trainers):
                agent.experience([obs_n[i], other_loc_n[i], other_idx_n[i], message_n[i], action_n[i], rew_n[i], new_obs_n[i], new_other_loc_n[i], new_other_idx_n[i],done_n[i]])
            obs_n = new_obs_n
            other_loc_n = new_other_loc_n
            other_idx_n = new_other_idx_n
            # get episode reward and comm freq
            comm_freq[-1] += num_comm
            num_comm = 0
            for i, rew in enumerate(rew_n):
                episode_rewards[-1] += rew/len(rew_n)
                agent_rewards[i][-1] += rew
            if done or terminal:
                comm_freq[-1] = comm_freq[-1]/(num_others*env.n_agents*arglist.max_episode_len)
                episode_rewards[-1]=episode_rewards[-1]/arglist.max_episode_len
                obs_n = env.reset()
                other_loc_n, other_idx_n = get_comm_pairs(obs_n, env.n_agents_obs, num_others)
                episode_step = 0
                comm_freq.append(0.0)
                episode_rewards.append(0.0)
                for a in agent_rewards:
                    a.append(0)
                agent_info.append([[]])
                # print(f'finish ep {training_step}')
            # increment global step counter
            training_step += 1

            # for benchmarking learned policies
            if arglist.benchmark:
                for i, info in enumerate(info_n):
                    agent_info[-1][i].append(info_n['n'])
                if train_step > arglist.benchmark_iters and (done or terminal):
                    file_name = arglist.benchmark_dir + arglist.exp_name + '.pkl'
                    print('Finished benchmarking, now saving...')
                    with open(file_name, 'wb') as fp:
                        pickle.dump(agent_info[:-1], fp)
                    break
                continue

            # for displaying learned policies
            if arglist.display:
                time.sleep(0.1)
                env.render()
            # update all trainers, if not in display or benchmark mode
            loss = None
            for agent in trainers:
                agent.preupdate()
            for agent in trainers:
                loss = agent.update(trainers, training_step)

            # save model, display training output
            if terminal and (len(episode_rewards) % arglist.save_rate == 0):
                mean_epi_reward = np.mean(episode_rewards[-arglist.save_rate:])
                mean_comm_freq = np.mean(comm_freq[-arglist.save_rate:])
                if mean_epi_reward > max_mean_epi_reward:
                    U.save_state(arglist.save_dir, saver=saver)
                    max_mean_epi_reward = mean_epi_reward
                    print("save checkpoint...")
                # print statement depends on whether or not there are adversaries
                if num_adversaries == 0:
                    print("steps: {}, episodes: {}, mean comm freq: {}, mean episode reward: {}, time: {}".format(
                        training_step, len(episode_rewards), mean_comm_freq, mean_epi_reward, round(time.time()-t_start, 3)))
                else:
                    print("steps: {}, episodes: {}, mean comm freq: {}, mean episode reward: {}, agent episode reward: {}, time: {}".format(
                        training_step, len(episode_rewards), mean_comm_freq, mean_epi_reward,
                        [np.mean(rew[-arglist.save_rate:]) for rew in agent_rewards], round(time.time()-t_start, 3)))
                t_start = time.time()
                # Keep track of final episode reward
                final_ep_rewards.append(np.mean(episode_rewards[-arglist.save_rate:]))
                for rew in agent_rewards:
                    final_ep_ag_rewards.append(np.mean(rew[-arglist.save_rate:]))

            # saves final episode reward for plotting training curve later
            if len(episode_rewards) > arglist.num_episodes:
                rew_file_name = arglist.plots_dir + arglist.exp_name + '_rewards.pkl'
                with open(rew_file_name, 'wb') as fp:
                    pickle.dump(final_ep_rewards, fp)
                agrew_file_name = arglist.plots_dir + arglist.exp_name + '_agrewards.pkl'
                with open(agrew_file_name, 'wb') as fp:
                    pickle.dump(final_ep_ag_rewards, fp)
                print('...Finished total of {} episodes.'.format(len(episode_rewards)))
                break

if __name__ == '__main__':
    arglist = parse_args()
    train(arglist)

#TODO: mod this file to train CMADDPG - m_model, get message etc. ref from I2C
#TODO: train CMADDG with IC3 PP env 

#TODO: then 1 agent:
# 1. get action from trained CMADDPG (first without input any message)
# 2. share obs and action OR encode obs and action before sending message 
# 3. input obs and messages then output action from CMADDPG 
# 4. dont forget to reward shaping

#TODO: train once or alternate?

#TODO: then 2 agent: 
# 1. get action from trained CMADDPG
# Internal loop of episode: 
#   2. get message from DMADDG (new MDP M' - e.g., get message from j and i send message to j)
# 3. output real action from CMADDPG conditioned on final messages from 2.
# 4. set reward for DMADDPG

#TODO: then 3 agent:  
# 1. get action from MADDPG 
# Internal loop of episode: 
#   2. get message from DMADDG 
# 3. output real action from CMADDPG conditioned on final messages from 2.
# 4. set reward for DMADDPG