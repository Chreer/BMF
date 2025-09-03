import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tools
import json

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class VAE(nn.Module):

    def __init__(self, input_dim, h_dim=512, z_dim=40):
        super(VAE, self).__init__()

        self.input_dim = input_dim
        self.h_dim = h_dim
        self.z_dim = z_dim

        self.fc1 = nn.Linear(input_dim, h_dim)  
        self.fc2 = nn.Linear(h_dim, z_dim)  # mu
        self.fc3 = nn.Linear(h_dim, z_dim)  # log_var

        self.fc4 = nn.Linear(z_dim, h_dim)
        self.fc5 = nn.Linear(h_dim, input_dim)

    def forward(self, x):
        """
        向前传播部分, 在model_name(inputs)时自动调用
        :param x: the input of our training model [b, batch_size, 1, 28, 28]
        :return: the result of our training model
        """
        batch_size = x.shape[0] 
        x = x.view(batch_size, self.input_dim)  

        # encoder
        mu, log_var = self.encode(x)
        # reparameterization trick
        sampled_z = self.reparameterization(mu, log_var)
        # decoder
        x_hat = self.decode(sampled_z)
        # reshape
        x_hat = x_hat.view(*x.shape)
        return x_hat, mu, log_var

    def encode(self, x):
        """
        encoding part
        :param x: input image
        :return: mu and log_var
        """
        h = F.relu(self.fc1(x))
        mu = self.fc2(h)
        log_var = self.fc3(h)

        return mu, log_var

    def reparameterization(self, mu, log_var):
        """
        Given a standard gaussian distribution epsilon ~ N(0,1),
        we can sample the random variable z as per z = mu + sigma * epsilon
        :param mu:
        :param log_var:
        :return: sampled z
        """
        sigma = torch.exp(log_var * 0.5)
        eps = torch.randn_like(sigma)
        return mu + sigma * eps 

    def decode(self, z):
        """
        Given a sampled z, decode it back to image
        :param z:
        :return:
        """
        h = F.leaky_relu(self.fc4(z))
        x_hat = self.fc5(h)
        return x_hat

class Base(nn.Module):
    """docstring for Base"""
    def __init__(self, view_space, feature_space, num_actions, hidden_size):
        super(Base, self).__init__()

        self.view_space = view_space  # view_width * view_height * n_channel
        self.feature_space = feature_space # feature_size
        self.num_actions = num_actions

        # for input_view
        self.l1 = nn.Linear(np.prod(view_space), hidden_size)
        # for input_feature
        self.l2 = nn.Linear(feature_space[0], hidden_size)
        # for input_act_prob
        self.l3 = nn.Linear(num_actions, 64)
        self.l4 = nn.Linear(64, 32)

    def forward(self, input_view, input_feature, input_act_prob):
        # flatten_view = torch.FloatTensor(input_view)
        flatten_view = input_view.reshape(-1, np.prod(self.view_space))
        h_view = F.relu(self.l1(flatten_view))

        h_emb  = F.relu(self.l2(input_feature))

        emb_prob = F.relu(self.l3(input_act_prob))
        dense_prob = F.relu(self.l4(emb_prob))

        concat_layer = torch.cat([h_view, h_emb, dense_prob], dim=1)
        return concat_layer

class Actor(nn.Module):
    """docstring for Actor"""
    def __init__(self, hidden_size, num_actions):
        super(Actor, self).__init__()

        self.l1 = nn.Linear(32 + 2 * hidden_size, hidden_size * 2)
        self.l2 = nn.Linear(hidden_size * 2, num_actions)

    def forward(self, concat_layer):
        dense = F.relu(self.l1(concat_layer))
        policy = F.softmax(self.l2(dense / 0.1), dim=-1)
        policy = torch.nan_to_num(policy,nan=0,posinf=1,neginf=-1)
        policy = policy.clamp(1e-10, 1-1e-10)
        return policy

class Critic(nn.Module):
    """docstring for Critic"""
    def __init__(self, hidden_size):
        super(Critic, self).__init__()

        self.l1 = nn.Linear(32 + 2 * hidden_size, hidden_size)
        self.l2 = nn.Linear(hidden_size, 1)

    def forward(self, concat_layer):
        dense = F.relu(self.l1(concat_layer))
        value = self.l2(dense)
        value = value.reshape(-1)
        return value

class VAE_MFAC:
    """docstring for MFAC"""
    def __init__(self, handle, env, value_coef=0.1, ent_coef=0.08, gamma=0.95, batch_size=64, learning_rate=1e-4):
        self.env = env

        self.view_space = env.unwrapped.env.get_view_space(handle)
        self.feature_space = env.unwrapped.env.get_feature_space(handle)
        self.num_actions = env.unwrapped.env.get_action_space(handle)[0]
        self.reward_decay = gamma

        self.batch_size = batch_size
        self.learning_rate = learning_rate

        self.value_coef = value_coef  # coefficient of value in the total loss
        self.ent_coef = ent_coef  # coefficient of entropy in the total loss

        # init training buffers
        self.replay_buffer = tools.EpisodesBuffer(use_mean=True)

        hidden_size = 256
        self.base = Base(self.view_space, self.feature_space, self.num_actions, hidden_size).to(device)
        self.actor = Actor(hidden_size, self.num_actions).to(device)
        self.critic = Critic(hidden_size).to(device)
        self.optimizer = torch.optim.Adam([
            {'params': self.base.parameters(),   'lr': learning_rate},
            {'params': self.actor.parameters(),  'lr': learning_rate},
            {'params': self.critic.parameters(), 'lr': learning_rate}
            ])
        
        self.actor_loss_trackor=list()
        self.critic_loss_trackor=list()
        self.ent_loss_trackor=list()
        
        self.vae_input = np.prod(self.view_space) + self.feature_space[0]
        self.vae_hidden = 512
        self.vae_representation = 32
        self.vae = VAE(self.vae_input, self.vae_hidden, self.vae_representation).to(device)
        self.vae_optimizer = torch.optim.Adam([
            {'params': self.vae.parameters(),   'lr': learning_rate}
            ])
        
        self.vae_loss_trackor=list()

    @property
    def vars(self):
        return [self.base, self.actor, self.critic]

    def act(self, **kwargs):
        input_view = torch.FloatTensor(kwargs['state'][0]).to(device)
        input_feature = torch.FloatTensor(kwargs['state'][1]).to(device)
        input_act_prob = torch.FloatTensor(kwargs['prob']).to(device)
        concat_layer = self.base(input_view, input_feature, input_act_prob)
        policy = self.actor(concat_layer)
        action = torch.multinomial(policy, 1)
        action = action.cpu().numpy()
        return action.astype(np.int32).reshape((-1,))

    def train(self, data_dir, is_buffer_clear=True):
        # calc buffer size
        n = 0
        # batch_data = sample_buffer.episodes()
        batch_data = self.replay_buffer.episodes()
        if is_buffer_clear: 
            self.replay_buffer = tools.EpisodesBuffer(use_mean=True)

        for episode in batch_data:
            n += len(episode.rewards)

        view = torch.FloatTensor(n, *self.view_space).to(device)
        feature = torch.FloatTensor(n, *self.feature_space).to(device)
        action = torch.LongTensor(n).to(device)
        reward = torch.FloatTensor(n).to(device)
        act_prob_buff = torch.FloatTensor(n, self.num_actions).to(device)

        ct = 0
        gamma = self.reward_decay
        # collect episodes from multiple separate buffers to a continuous buffer
        for k, episode in enumerate(batch_data):
            v, f, a, r, prob = episode.views, episode.features, episode.actions, episode.rewards, episode.probs
            v = torch.FloatTensor(v).to(device)
            f = torch.FloatTensor(f).to(device)
            r = torch.FloatTensor(r).to(device)
            a = torch.LongTensor(a).to(device)
            prob = torch.FloatTensor(prob).to(device)

            m = len(episode.rewards)
            assert len(episode.probs) > 0

            concat_layer = self.base(v[-1].reshape(1, -1), f[-1].reshape(1, -1), prob[-1].reshape(1, -1))
            keep = self.critic(concat_layer)[0]

            for i in reversed(range(m)):
                keep = keep * gamma + r[i]
                r[i] = keep

            view[ct:ct + m] = v
            feature[ct:ct + m] = f
            action[ct:ct + m] = a
            reward[ct:ct + m] = r
            act_prob_buff[ct:ct + m] = prob
            ct += m

        assert n == ct

        # train
        concat_layer = self.base(view, feature, act_prob_buff)
        value = self.critic(concat_layer)
        policy = self.actor(concat_layer)

        action_mask = F.one_hot(action, self.num_actions)
        advantage = (reward - value).detach()

        log_policy = torch.log(policy + 1e-6)
        log_prob = torch.sum(log_policy * action_mask, dim=1)

        pg_loss = -torch.mean(advantage * log_prob)
        vf_loss = self.value_coef * torch.mean(torch.square(reward.detach() - value))
        neg_entropy = self.ent_coef * torch.mean(torch.sum(policy * log_policy, dim=1))
        total_loss = pg_loss + vf_loss + neg_entropy

        # train op (clip gradient)
        self.optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(self.base.parameters(), 0.5)
        nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
        nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
        self.optimizer.step()
        
        self.critic_loss_trackor.append(np.round(vf_loss.item(), 6))
        self.actor_loss_trackor.append(np.round(pg_loss.item(), 6))
        self.ent_loss_trackor.append(np.round(neg_entropy.item(), 6))
        
        with open(os.path.join(data_dir,'critic_loss.json'),'w') as f:
            json.dump(str(self.critic_loss_trackor),f)
        with open(os.path.join(data_dir,'actor_loss.json'),'w') as f:
            json.dump(str(self.actor_loss_trackor),f)
        with open(os.path.join(data_dir,'neg_ent_loss.json'),'w') as f:
            json.dump(str(self.ent_loss_trackor),f)
        

        print('[*] PG_LOSS:', np.round(pg_loss.item(), 6), '/ VF_LOSS:', np.round(vf_loss.item(), 6), '/ ENT_LOSS:', np.round(neg_entropy.item(), 6), '/ VALUE:', np.mean(value.cpu().detach().numpy()))

    def flush_buffer(self, **kwargs):
        self.replay_buffer.push(**kwargs)

    def save(self, dir_path, step=0):
        os.makedirs(dir_path, exist_ok=True)

        model_vars = {
            'base':   self.base.state_dict(),
            'actor':  self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'vae': self.vae.state_dict()
        }
        
        step = 0 # for less space

        file_path = os.path.join(dir_path, "mfac_{}.pth".format(0))
        torch.save(model_vars, file_path)

        print("[*] Model saved at: {}".format(file_path))

    def load(self, dir_path, step=0):
        file_path = os.path.join(dir_path, "mfac_{}.pth".format(0))
        model_vars = torch.load(file_path)

        self.base.load_state_dict(model_vars['base'])
        self.actor.load_state_dict(model_vars['actor'])
        self.critic.load_state_dict(model_vars['critic'])
        self.vae.load_state_dict(model_vars['vae'])

        print("[*] Loaded model from {}".format(file_path))
        
    def load_vae(self, dir_path, step=0):
        file_path = os.path.join(dir_path, "mfac_{}.pth".format(0))
        model_vars = torch.load(file_path)

        self.vae.load_state_dict(model_vars['vae'])

        print("[*] Loaded vae model from {}".format(file_path))

    def vae_train(self, data_dir):
        # calc buffer size
        n = 0
        # batch_data = sample_buffer.episodes()
        batch_data = self.replay_buffer.episodes()
        self.replay_buffer = tools.EpisodesBuffer(use_mean=True)

        for episode in batch_data:
            n += len(episode.rewards)

        view = torch.FloatTensor(n, *self.view_space).to(device)
        feature = torch.FloatTensor(n, *self.feature_space).to(device)

        ct = 0
        # collect episodes from multiple separate buffers to a continuous buffer
        for k, episode in enumerate(batch_data):
            v, f, a, r, prob = episode.views, episode.features, episode.actions, episode.rewards, episode.probs
            v = torch.FloatTensor(v).to(device)
            f = torch.FloatTensor(f).to(device)

            m = len(episode.rewards)

            view[ct:ct + m] = v
            feature[ct:ct + m] = f
            ct += m

        assert n == ct

        # train
        flatten_view = view.reshape(-1, np.prod(self.view_space))
        
        x = torch.cat([flatten_view, feature], dim=1)

        x_hat, mu, log_var = self.vae(x)

        vae_loss = F.mse_loss(x_hat, x, reduction='mean')+0.5 * torch.sum(torch.exp(log_var) + torch.pow(mu, 2) - 1. - log_var)

        self.vae_optimizer.zero_grad()
        vae_loss.backward()
        self.vae_optimizer.step()
        
        self.vae_loss_trackor.append(np.round(vae_loss.item(), 6))
        
        with open(os.path.join(data_dir,'vae_loss.json'),'w') as f:
            json.dump(str(self.vae_loss_trackor),f)

        
    def encode_agent(self, **kwargs):
        input_view = torch.FloatTensor(kwargs['state'][0]).to(device)
        input_feature = torch.FloatTensor(kwargs['state'][1]).to(device)
        flatten_view = input_view.reshape(-1, np.prod(self.view_space))
        x = torch.cat([flatten_view, input_feature], dim=1)
        mu, log_var = self.vae.encode(x)
        sampled_z = self.vae.reparameterization(mu, log_var)
        return sampled_z.cpu().detach().numpy()