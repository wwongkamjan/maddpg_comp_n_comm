import numpy as np
import random
import tensorflow as tf
import maddpg.common.tf_util as U

from maddpg.common.distributions import make_pdtype
from maddpg import AgentTrainer
from maddpg.trainer.replay_buffer import ReplayBuffer


def discount_with_dones(rewards, dones, gamma):
    discounted = []
    r = 0
    for reward, done in zip(rewards[::-1], dones[::-1]):
        r = reward + gamma*r
        r = r*(1.-done)
        discounted.append(r)
    return discounted[::-1]

def make_update_exp(vals, target_vals):
    polyak = 1.0 - 1e-2
    expression = []
    for var, var_target in zip(sorted(vals, key=lambda v: v.name), sorted(target_vals, key=lambda v: v.name)):
        expression.append(var_target.assign(polyak * var_target + (1.0-polyak) * var))
    expression = tf.group(*expression)
    return U.function([], [], updates=[expression])

def p_train(make_obs_ph_n, make_message_ph_n, act_space_n, num_agents_obs, p_index, m_func, p_func, q_func, optimizer, grad_norm_clipping=None, local_q_func=False, num_units=64, scope="trainer", reuse=tf.AUTO_REUSE):
    with tf.variable_scope(scope, reuse=reuse):
        # print('in p train')
        # create distribtuions
        act_pdtype_n = [make_pdtype(act_space) for act_space in act_space_n]

        # set up placeholders
        obs_ph_n = make_obs_ph_n
        message_ph_n = make_message_ph_n
        act_ph_n = [act_pdtype_n[i].sample_placeholder([None], name="action"+str(i)) for i in range(len(act_space_n))]

        m_input = message_ph_n[p_index]
        encode_dim = m_input.get_shape().as_list()[-1]

        # message encoder
        message_encode = m_func(m_input,encode_dim, num_agents_obs, scope='m_func', num_units=num_units)
        m_func_vars = U.scope_vars(U.absolute_scope_name("m_func"))

        # policy
        p_input = tf.concat((obs_ph_n[p_index], message_encode), 1)
        p = p_func(p_input, int(act_pdtype_n[p_index].param_shape()[0]), scope="p_func", num_units=num_units)
        p_func_vars = U.scope_vars(U.absolute_scope_name("p_func"))

        # wrap parameters in distribution
        act_pd = act_pdtype_n[p_index].pdfromflat(p)

        act_sample = act_pd.sample()
        p_reg = tf.reduce_mean(tf.square(act_pd.flatparam()))

        act_input_n = act_ph_n + []
        act_input_n[p_index] = act_pd.sample()

        # q_obs_acts = tf.concat(obs_ph_n + act_ph_n, 1)
        # q_input = tf.concat((q_obs_acts, message_encode), 1) 
        q_input = tf.concat(obs_ph_n + act_input_n, 1)

        if local_q_func:
            q_input = tf.concat([obs_ph_n[p_index], act_input_n[p_index]], 1)
        q = q_func(q_input, 1, scope="q_func", reuse=True, num_units=num_units)[:,0]
        
        # loss and optimization
        pg_loss = -tf.reduce_mean(q)
        loss = pg_loss + p_reg * 1e-3
        optimize_expr = U.minimize_and_clip(optimizer, loss, [p_func_vars,m_func_vars], grad_norm_clipping)

        # Create callable functions
        train = U.function(inputs=obs_ph_n + message_ph_n + act_ph_n, outputs= loss, updates=[optimize_expr])
        act = U.function(inputs=[obs_ph_n[p_index], message_ph_n[p_index]], outputs=act_sample)
        p_values = U.function([obs_ph_n[p_index], message_ph_n[p_index]], outputs=p)

        # target network
        target_message_encode = m_func(m_input, encode_dim, num_agents_obs, scope='target_m_func', num_units=num_units)
        target_m_func_vars = U.scope_vars(U.absolute_scope_name("target_m_func"))

        p_input = tf.concat((obs_ph_n[p_index], target_message_encode), 1)
        target_p = p_func(p_input, int(act_pdtype_n[p_index].param_shape()[0]), scope="target_p_func", num_units=num_units)
        target_p_func_vars = U.scope_vars(U.absolute_scope_name("target_p_func"))

        update_target_p = make_update_exp(p_func_vars, target_p_func_vars)
        update_target_m = make_update_exp(m_func_vars, target_m_func_vars)

        target_act_sample = act_pdtype_n[p_index].pdfromflat(target_p).sample()
        target_act = U.function(inputs=[obs_ph_n[p_index], message_ph_n[p_index]], outputs=target_act_sample)

        return act, train, update_target_p, update_target_m, {'p_values': p_values, 'target_act': target_act}

# def q_train(make_obs_ph_n, make_message_ph_n, act_space_n, num_agents_obs, q_index, m_func, q_func, optimizer, grad_norm_clipping=None, local_q_func=False, scope="trainer", reuse=tf.AUTO_REUSE, num_units=64):
def q_train(make_obs_ph_n, act_space_n, q_index, q_func, optimizer, grad_norm_clipping=None, local_q_func=False, scope="trainer", reuse=tf.AUTO_REUSE, num_units=64):
    with tf.variable_scope(scope, reuse=reuse):
        # create distribtuions
        act_pdtype_n = [make_pdtype(act_space) for act_space in act_space_n]

        # set up placeholders
        obs_ph_n = make_obs_ph_n
        # message_ph_n = make_message_ph_n

        # m_input = message_ph_n[q_index]
        # encode_dim = m_input.get_shape().as_list()[-1]

        # # message encoder
        # # print('in q train')
        # message_encode = m_func(m_input,encode_dim, num_agents_obs, scope='m_func', num_units=num_units)
        # m_func_vars = U.scope_vars(U.absolute_scope_name("m_func"))

        act_ph_n = [act_pdtype_n[i].sample_placeholder([None], name="action"+str(i)) for i in range(len(act_space_n))]
        target_ph = tf.placeholder(tf.float32, [None], name="target")
        # print(obs_ph_n)
        # print(act_ph_n)
        # q_obs_acts = tf.concat(obs_ph_n + act_ph_n, 1)
        # q_input = tf.concat((q_obs_acts, message_encode), 1) 
        # print(q_input)
        q_input = tf.concat(obs_ph_n + act_ph_n, 1)
        if local_q_func:
            q_input = tf.concat([obs_ph_n[q_index], act_ph_n[q_index]], 1)
        q = q_func(q_input, 1, scope="q_func", num_units=num_units)[:,0]
        q_func_vars = U.scope_vars(U.absolute_scope_name("q_func"))

        q_loss = tf.reduce_mean(tf.square(q - target_ph))

        # viscosity solution to Bellman differential equation in place of an initial condition
        q_reg = tf.reduce_mean(tf.square(q))
        loss = q_loss #+ 1e-3 * q_reg

        optimize_expr = U.minimize_and_clip(optimizer, loss, q_func_vars, grad_norm_clipping)

        # Create callable functions
        train = U.function(inputs=obs_ph_n + act_ph_n + [target_ph], outputs=loss, updates=[optimize_expr])
        q_values = U.function(obs_ph_n + act_ph_n, q)
        # train = U.function(inputs=obs_ph_n + act_ph_n + message_ph_n + [target_ph], outputs=loss, updates=[optimize_expr])
        # q_values = U.function(obs_ph_n + act_ph_n + message_ph_n, q)

        # target_message_encode = m_func(m_input, encode_dim, num_agents_obs, scope='target_m_func', num_units=num_units)
        # target_m_func_vars = U.scope_vars(U.absolute_scope_name("target_m_func"))

        # q_obs_acts = tf.concat(obs_ph_n + act_ph_n, 1)
        # q_input = tf.concat((q_obs_acts, target_message_encode), 1) 

        # target network
        target_q = q_func(q_input, 1, scope="target_q_func", num_units=num_units)[:,0]
        target_q_func_vars = U.scope_vars(U.absolute_scope_name("target_q_func"))
        update_target_q = make_update_exp(q_func_vars, target_q_func_vars)

        # update_target_m = make_update_exp(m_func_vars, target_m_func_vars)

        # target_q_values = U.function(obs_ph_n + act_ph_n + message_ph_n, target_q)

        # return train, update_target_q, update_target_m, {'q_values': q_values, 'target_q_values': target_q_values}
        target_q_values = U.function(obs_ph_n + act_ph_n, target_q)

        return train, update_target_q, {'q_values': q_values, 'target_q_values': target_q_values}

class CMADDPGAgentTrainer(AgentTrainer):
    def __init__(self, name, model, obs_shape_n, message_shape_n, target_loc_space_n, act_space_n, agent_index, num_agents_obs, args, local_q_func=False):
        self.name = name
        self.n = len(obs_shape_n)
        self.agent_index = agent_index
        self.args = args
        obs_ph_n = []
        message_ph_n = []
        target_loc_ph_n = []
        for i in range(self.n):
            obs_ph_n.append(U.BatchInput(obs_shape_n[i], name="observation"+str(i)).get())
            message_ph_n.append(U.BatchInput(message_shape_n[i], name="message"+str(i)).get())
            target_loc_ph_n.append(U.BatchInput(target_loc_space_n[i], name="target_location"+str(i)).get())
        self.num_agents_obs = num_agents_obs
        self.model = model
        self.obs_ph_n = obs_ph_n
        self.message_ph_n = message_ph_n
        self.target_loc_ph_n = target_loc_ph_n
        self.local_q_func = local_q_func
        self.act_space_n = act_space_n  
        # Create experience buffer
        self.replay_buffer_general = ReplayBuffer(1e6)
        self.max_replay_buffer_len = args.batch_size * args.max_episode_len
        self.replay_sample_index = None
        self.act = None
        self.p_train = None
        self.p_update =None
        self.m_update=None
        self.p_m_debug=None
        self.q_train=None
        self.q_update=None
        self.q_debug=None
        self.step = 0

    # Create all the functions necessary to train the model
    def initial_q_model(self):
        # self.q_train, self.q_update, self.m_update ,self.q_debug = q_train(
        #     scope=self.name,
        #     make_obs_ph_n=self.obs_ph_n,
        #     make_message_ph_n = self.message_ph_n,
        #     act_space_n=self.act_space_n,
        #     num_agents_obs= self.num_agents_obs,
        #     q_index=self.agent_index,
        #     m_func=self.model[1],
        #     q_func=self.model[0],
        #     optimizer=tf.train.AdamOptimizer(learning_rate=self.args.lr),
        #     grad_norm_clipping=0.5,
        #     local_q_func=self.local_q_func,
        #     num_units=self.args.num_units
        # )
                # Create all the functions necessary to train the model
        self.q_train, self.q_update, self.q_debug = q_train(
            scope=self.name,
            make_obs_ph_n=self.obs_ph_n,
            act_space_n=self.act_space_n,
            q_index=self.agent_index,
            q_func=self.model[0],
            optimizer=tf.train.AdamOptimizer(learning_rate=self.args.lr),
            grad_norm_clipping=0.5,
            local_q_func=self.local_q_func,
            num_units=self.args.num_units
        )

    def initial_p_model(self):
        self.act, self.p_train, self.p_update, self.m_update, self.p_debug = p_train(
            scope=self.name,
            make_obs_ph_n=self.obs_ph_n,
            make_message_ph_n = self.message_ph_n,
            act_space_n=self.act_space_n,
            num_agents_obs= self.num_agents_obs,
            p_index=self.agent_index,
            m_func=self.model[1],
            p_func=self.model[0],
            q_func=self.model[0],
            optimizer=tf.train.AdamOptimizer(learning_rate=self.args.lr),
            grad_norm_clipping=0.5,
            local_q_func=self.local_q_func,
            num_units=self.args.num_units
        )
    def target_comm(self, obs, target_loc):
        return True

    def action(self, obs, message):
        return self.act(obs[None], message[None])[0]

    def experience(self, data):#obs, message, act, rew, new_obs, done, terminal):
        # Store transition in the replay buffer. 
        self.replay_buffer_general.add(data) #obs, message, act, rew, new_obs, float(done)

    def preupdate(self):
        self.replay_sample_index = None

    def update(self, agents, t):
        if len(self.replay_buffer_general) < self.max_replay_buffer_len: # replay buffer is not large enough
            return
        if not t % 100 == 0:  # only update every 100 steps
            return 
        # collect replay sample from all agents
        batch_size = self.args.batch_size
        self.replay_sample_index = self.replay_buffer_general.make_index(batch_size) 
        obs_n = []
        obs_next_n = []
        act_n = []
        message_n = []
        target_loc_next_n = []
        target_idx_next_n = []
        index = self.replay_sample_index
        for i in range(self.n):
            obs, target_loc, target_idx, message, act, rew, obs_next, target_loc_next, target_idx_next, done = agents[i].replay_buffer_general.sample_index(index)
            obs_n.append(obs)
            obs_next_n.append(obs_next)
            message_n.append(message)
            act_n.append(act)
            target_loc_next_n.append(target_loc_next)
            target_idx_next_n.append(target_idx_next)
        obs, target_loc, target_idx, message, act, rew, obs_next, target_loc_next, target_idx_next, done = self.replay_buffer_general.sample_index(index)
        # shape of target_loc_next (batch_size, num_agents_obs, obs_dim)
        num_agents_obs = self.num_agents_obs
        message_next_n = [np.zeros((batch_size, num_agents_obs, len(obs_next[0]))) for i in range(self.n)]
        # get message for next step
        flags_n_tmp = []
        for i in range(self.n):
            flags_tmp = []
            for j in range(num_agents_obs):
                flags_tmp.append([[True for i in range(num_agents_obs)] for j in range(batch_size)])
            flags_n_tmp.append(flags_tmp)
        for i in range(batch_size):
            for j in range(self.n):
                for k in range(num_agents_obs):
                        target_idx = target_idx_next_n[j][i,k]
                        idx_tmp = 0
                        if flags_n_tmp[j][k][i] == True:
                            message_next_n[j][i, idx_tmp, :] = obs_next_n[target_idx][i,:]
                            idx_tmp = idx_tmp + 1
        # train q network
        num_sample = 1
        target_q = 0.0
        for j in range(num_sample):
            target_act_next_n = [agents[i].p_debug['target_act'](*([obs_next_n[i]]+[message_next_n[i]])) for i in range(self.n)]
            # target_q_next = self.q_debug['target_q_values'](*(obs_next_n + target_act_next_n + message_next_n))
            target_q_next = self.q_debug['target_q_values'](*(obs_next_n + target_act_next_n))
            target_q += rew + self.args.gamma * (1.0 - done) * target_q_next
        target_q /= num_sample
        q_loss = self.q_train(*(obs_n + act_n  + [target_q]))
        # train p and m network
        p_loss = self.p_train(*(obs_n + message_n + act_n))
        # c_loss = None   
        # update p, q target network   
        self.p_update()
        self.m_update()
        self.q_update()
        return [q_loss, p_loss, np.mean(target_q), np.mean(rew), np.mean(target_q_next), np.std(target_q)]
