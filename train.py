
import torch
import os
import yaml
import random
import numpy as np
import time
import copy
import json
from tqdm import tqdm, trange
import math
import tools
import heapq
from pettingzoo.magent import battle_v3
from sklearn.metrics.pairwise import cosine_similarity
from sklearn import metrics
import vae_networks
VAE_MFAC = vae_networks.VAE_MFAC

from joblib import Parallel, delayed, parallel_backend
from sklearn.cluster import KMeans


class Color:
    INFO = '\033[1;34m{}\033[0m'
    WARNING = '\033[1;33m{}\033[0m'
    ERROR = '\033[1;31m{}\033[0m'
class MARL(object):
    def __init__(self,
                map_size=10,
                max_steps=10,
                max_episode=100000,
                render=False,
                update_interval=100,
                save_interval=1000,
                buffer_capacity=50000,
                a_coe=0.1,
                c_coe = 0.1,
                ent_coe=0.08,
                seed_k=233,
                
                num_classes = 2,
                vae_train = False,
                group_interval = 50
                ):
        super().__init__()

        print('Initializing...')

        # generate config
        config_data=locals()
        del config_data['self']
        del config_data['__class__']
        time_data=time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime())
        config_data['time']=time_data

        # environment
        self.device='cuda' if torch.cuda.is_available() else 'cpu'
        torch.manual_seed(seed_k)
        if self.device=='cuda':
            torch.cuda.manual_seed(seed_k)
        np.random.seed(seed_k)
        self.seed_k = seed_k

        # args
        self.map_size=map_size
        self.max_steps=max_steps
        self.max_episode=max_episode
        # self.vae_max_episode=max_episode//2
        self.vae_max_episode=2000
        self.vae_train = vae_train
        self.render=render
        self.update_interval=update_interval
        self.save_interval=save_interval
        self.buffer_capacity=buffer_capacity
        self.a_coe = a_coe
        self.c_coe = c_coe
        self.ent_coe = ent_coe
        
        self.num_classes = num_classes
        self.group_interval = group_interval

        # making output directory
        curfile_name = os.path.basename(__file__)
        curfile_name = curfile_name.split('.')[0]
        
        self.model_dir=os.path.join('.','model',f'{self.map_size*self.map_size}grid_{time_data}')
        self.data_dir=os.path.join('.','data',f'{self.map_size*self.map_size}grid_{time_data}')  
        self.render_dir=os.path.join('.','render',f'{self.map_size*self.map_size}grid_{time_data}')  
        self.vae_model_dir=os.path.join('.','vae_model',f'{self.map_size*self.map_size}grid_{time_data}')
        self.vae_data_dir=os.path.join('.','vae_data',f'{self.map_size*self.map_size}grid_{time_data}')  
        self.vae_vis_dir=os.path.join('.','vae_data',f'{self.map_size*self.map_size}grid_{time_data}')  
        os.mkdir(self.model_dir)
        os.mkdir(self.data_dir)
        os.mkdir(self.render_dir)
        os.mkdir(self.vae_model_dir)
        os.mkdir(self.vae_data_dir)
        os.mkdir(self.vae_vis_dir)
        
        with open(os.path.join(self.data_dir,'config.json'),'w') as f:
            json.dump(config_data,f)

        print(f'Training platform with {self.map_size*self.map_size} grids initialized')
    
    def linear_decay(self, epoch, x, y):
        min_v, max_v = y[0], y[-1]
        start, end = x[0], x[-1]

        if epoch == start:
            return min_v

        eps = min_v

        for i, x_i in enumerate(x):
            if epoch <= x_i:
                interval = (y[i] - y[i - 1]) / (x_i - x[i - 1])
                eps = interval * (epoch - x[i - 1]) + y[i - 1]
                break

        return eps
    
    def vae_spawn_ai(self, env, handle):
        model = VAE_MFAC(handle, env)
        return model
    
    def construct_agent_net(self, state, ids):
        
        # state_feature1 = state[0][group_index_list].reshape(alive_num, -1)
        # state_feature2 = state[1][group_index_list]
        # state_feature = np.concatenate((state_feature1, state_feature2),axis=-1)
        state_cossim = cosine_similarity(state)
        
        # action_feature = acts.reshape(alive_num,-1)
        # action_cossim = cosine_similarity(action_feature) 
        
        # np.fill_diagonal(state_cossim, 0)
        state_cossim = state_cossim/state_cossim.sum(axis=1).reshape(-1,1)
        
        return state_cossim
    
    def mutual_info(self, x, y):
        return metrics.mutual_info_score(x,y)  
    
    def para_mutual_info_list(self, x, main_feature, n_threads = 1):
        n_threads = self.num_classes
        with parallel_backend("loky", inner_max_num_threads=n_threads):
            out2 = Parallel(n_jobs=n_threads)(delayed(self.mutual_info)(x, y) for y in main_feature)
        # self.mutual_info_list[idx] = np.array(out2).sum()
        return np.array(out2)
    
    
    def vae_group(self, sample_z, ids):
        
        alive_num = len(ids)
        
        cluster = KMeans(n_clusters= self.num_classes)
        cluster = cluster.fit(sample_z)
        group_index = cluster.labels_
        
        return group_index
    
        
    def vae_group_vis(self, sample_z):
        
        cluster = KMeans(n_clusters= self.num_classes)
        cluster = cluster.fit(sample_z)
        group_index = cluster.labels_
        
        return group_index
    
    def cul_action_all(self, sample_z_intra, action_intra):
        
        
        
        state_cossim = cosine_similarity(sample_z_intra)
    
        weight = state_cossim/state_cossim.sum(axis=1).reshape(-1,1)
        
        action_all = weight @ action_intra
        
        return action_all
    
    def mask_former_act_prob(self, former_act_prob, ids_former, ids):
        ids_former_one_hot = np.eye(len(ids_former))
        die_ids = np.setdiff1d(ids_former, ids)
        die_ids_idx = np.where(np.in1d(ids_former, die_ids))
        ids_former_one_hot = np.delete(ids_former_one_hot, die_ids_idx, axis=0)
        
        former_act_prob = ids_former_one_hot @ former_act_prob
        
        return former_act_prob
    
    def mask_group_init(self, group_init, ids_former, ids):
        ids_former_one_hot = np.eye(len(ids_former))
        die_ids = np.setdiff1d(ids_former, ids)
        die_ids_idx = np.where(np.in1d(ids_former, die_ids))
        ids_former_one_hot = np.delete(ids_former_one_hot, die_ids_idx, axis=0)
        
        group = ids_former_one_hot @ group_init
        
        return group
    
    def play(self, env, n_round, map_size, max_steps, handles, models, print_every, eps=1.0, render=False, train=False, vae_train=False):
        """play a ground and train"""
        env.reset()
        # generate_map(env, map_size, handles)

        step_ct = 0
        done = False

        obs_list = []
        if render:
            obs_list.append(env.render(mode='rgb_array'))

        n_group = len(handles)
        state = [None for _ in range(n_group)]
        acts = [None for _ in range(n_group)]
        ids = [None for _ in range(n_group)]
        ids_former = [None for _ in range(n_group)]
        group_ids_former = [None for _ in range(n_group)]

        alives = [None for _ in range(n_group)]
        rewards = [None for _ in range(n_group)]
        nums = [env.unwrapped.env.get_num(handle) for handle in handles]
        max_nums = nums.copy()

        loss = [None for _ in range(n_group)]
        eval_q = [None for _ in range(n_group)]
        n_action = [env.unwrapped.env.get_action_space(handles[0])[0], env.unwrapped.env.get_action_space(handles[1])[0]]

        print("\n\n[*] ROUND #{0}, EPS: {1:.2f} NUMBER: {2}".format(n_round, eps, nums))
        mean_rewards = [[] for _ in range(n_group)]
        total_rewards = [[] for _ in range(n_group)]

        former_act_prob = [np.zeros((max_nums[0], env.unwrapped.env.get_action_space(handles[0])[0])), np.zeros((max_nums[1], env.unwrapped.env.get_action_space(handles[1])[0]))]
        
        group_init = [None for _ in range(n_group)]

        # 存储前一时刻存活的智能体用于修复former_act_prob
        for i in range(n_group):
            ids_former[i] = env.unwrapped.env.get_agent_id(handles[i])
            
        while not done and step_ct < max_steps:
            # take actions for every model
            for i in range(n_group):
                state[i] = list(env.unwrapped.env.get_observation(handles[i]))
                ids[i] = env.unwrapped.env.get_agent_id(handles[i])

            for i in range(n_group):       
                former_act_prob[i]=self.mask_former_act_prob(former_act_prob[i],ids_former[i],ids[i])
                acts[i] = models[i].act(state=state[i], prob=former_act_prob[i], eps=eps)
                
                ids_former[i] = ids[i]

            for i in range(n_group):
                env.unwrapped.env.set_action(handles[i], acts[i])

            # simulate one step
            done = env.unwrapped.env.step()

            for i in range(n_group):
                rewards[i] = env.unwrapped.env.get_reward(handles[i])
                alives[i] = env.unwrapped.env.get_alive(handles[i])

            buffer = {
                'state': state[0], 'acts': acts[0], 'rewards': rewards[0],
                'alives': alives[0], 'ids': ids[0]
            }

            buffer['prob'] = former_act_prob[0]

            for i in range(n_group):
                sampled_z = models[i].encode_agent(state=state[i])
                if len(ids[i]) <= self.num_classes:
                    weight = self.construct_agent_net(sampled_z,ids[i])
                    action_one_hot = np.array(list(map(lambda x: np.eye(n_action[i])[x], acts[i])))
                    former_act_prob[i] = weight @ action_one_hot
                else:
                    if step_ct%self.group_interval == 0:
                        group_init[i] = self.vae_group(sampled_z,ids[i])
                        group_ids_former[i] = ids[i]
                    group = self.mask_group_init(group_init[i], group_ids_former[i], ids[i])
                    group_index = [None for _ in range(self.num_classes)]
                
                    for j in range(self.num_classes):
                        group_index[j] = np.argwhere(group==j).reshape(-1)
                    
                    action_one_hot = np.array(list(map(lambda x: np.eye(n_action[i])[x], acts[i])))
                    action_gmf = action_one_hot
                    none_action = np.zeros_like(action_one_hot[0])
                    none_sampled_z = np.zeros_like(sampled_z[0])
                    action_intra = [none_action for _ in range(self.num_classes)]
                    sampled_z_intra = [none_sampled_z for _ in range(self.num_classes)]
                    
                    for k in range(self.num_classes):
                        if len(group_index[k]) > 0:
                            action_intra[k] = np.mean(action_one_hot[group_index[k]],axis=0)
                            sampled_z_intra[k] = np.mean(sampled_z[[group_index[k]]],axis=0)
                    action_all = self.cul_action_all(sampled_z_intra, action_intra)
                    
                    for m in range(self.num_classes):
                        if len(group_index[m]) > 0:
                            action_gmf[group_index[m]] = action_all[m]
                    former_act_prob[i] = action_gmf

            if train or vae_train:
                models[0].flush_buffer(**buffer)

            if render:
                obs_list.append(env.render(mode='rgb_array'))

            # clear dead agents
            env.unwrapped.env.clear_dead()
            # stat info
            nums = [env.unwrapped.env.get_num(handle) for handle in handles]

            for i in range(n_group):
                sum_reward = sum(rewards[i])
                rewards[i] = sum_reward / nums[i]
                mean_rewards[i].append(rewards[i])
                total_rewards[i].append(sum_reward)


            info = {"Ave-Reward": np.round(rewards, decimals=6), "NUM": nums}

            step_ct += 1

            if step_ct % print_every == 0:
                print("> step #{}, info: {}".format(step_ct, info))

        if train:
            models[0].train(self.data_dir, True)
        if vae_train:
            models[0].train(self.data_dir, False)
            models[0].vae_train(self.vae_data_dir)

        for i in range(n_group):
            mean_rewards[i] = sum(mean_rewards[i]) / len(mean_rewards[i])
            total_rewards[i] = sum(total_rewards[i])

        return max_nums, nums, mean_rewards, total_rewards, obs_list

    def train(self):
        # Initialize the environment
        env = battle_v3.env(
            map_size=self.map_size,
            minimap_mode=True,
            step_reward=-0.005,
            dead_penalty=-0.5,
            attack_penalty=-0.1,
            attack_opponent_reward=0.5,
            max_cycles=self.max_steps,
            extra_features=True
        )
        handles = env.unwrapped.env.get_handles()
        
        log_dir = os.path.join(self.data_dir, 'log')
        render_dir = self.render_dir
        model_dir = self.model_dir

        models = [self.vae_spawn_ai(env, handles[0]), self.vae_spawn_ai(env, handles[1])]
        
        if not self.vae_train:
            self.vae_model_load_dir=os.path.join('.','vae_model','use')
            models[0].load_vae(self.vae_model_load_dir)
            models[1].load_vae(self.vae_model_load_dir)
        
        runner = tools.Runner(env, handles,self.map_size, self.max_steps, models, self.play, render=self.render, save_every=self.save_interval, tau=0.01, log_name='ac',
            log_dir=log_dir, model_dir=model_dir, render_dir=render_dir, train=True, vae_train=self.vae_train)

        eps = 1.0
        if self.vae_train:
            for vk in trange(0, self.vae_max_episode):
                runner.run_vae(eps, vk, self.data_dir)
        
        for k in trange(0, self.max_episode):
            eps = self.linear_decay(k, [0, int(self.max_episode * 0.8), self.max_episode], [1, 0.2, 0.1])
            runner.run(eps, k+self.vae_max_episode, self.data_dir)       

if __name__ == '__main__':
    
    # read config
    with open("config.yaml", encoding='utf-8')as file: content = file.read()
    config = yaml.load(content, Loader=yaml.FullLoader)
    
    with torch.cuda.device(config['device_id']):
        
        seed = random.randint(1, 10000)
        
        train_platform=MARL(map_size = config["train_args"]["map_size"],
                            max_steps = config["train_args"]["max_steps"],
                            max_episode = config["train_args"]["max_episode"],
                            render = config["train_args"]["render"],
                            save_interval=config["train_args"]["save_interval"],
                            update_interval=config["train_args"]["update_interval"],
                            buffer_capacity = config["train_args"]["buffer_capacity"],
                            a_coe = config["train_args"]["actor_coefficient"],
                            c_coe = config["train_args"]["critic_coefficient"],
                            ent_coe = config["train_args"]["entropy_coefficient"],
                            
                            num_classes=config["train_args"]["num_classes"],
                            vae_train=config["train_args"]["vae_train"],
                            
                            seed_k = seed)
        
        train_platform.train()