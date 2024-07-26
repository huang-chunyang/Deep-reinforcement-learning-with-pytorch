import argparse
import pickle
from collections import namedtuple

import matplotlib.pyplot as plt

import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal, Categorical
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

parser = argparse.ArgumentParser(description='Solve the Cartpole-v1 with PPO')
parser.add_argument(
    '--gamma', type=float, default=0.95, metavar='G', help='discount factor (default: 0.9)')
parser.add_argument('--seed', type=int, default=0, metavar='N', help='random seed (default: 0)')
parser.add_argument('--render', action='store_true', help='render the environment')
parser.add_argument(
    '--log-interval',
    type=int,
    default=10,
    metavar='N',
    help='interval between training status logs (default: 10)')
args = parser.parse_args()

torch.manual_seed(args.seed)

TrainingRecord = namedtuple('TrainingRecord', ['ep', 'reward'])
Transition = namedtuple('Transition', ['s', 'a', 'a_log_p', 'r', 's_'])


class ActorNet(nn.Module):

    def __init__(self):
        super(ActorNet, self).__init__()
        self.fc1 = nn.Linear(4, 128)
        self.fc2 = nn.Linear(128, 2) 

    def forward(self, x):
        x = F.relu(self.fc1(x))
        action_probs = F.softmax(self.fc2(x), dim=1)
        return action_probs


class CriticNet(nn.Module):

    def __init__(self):
        super(CriticNet, self).__init__()
        self.fc = nn.Linear(4, 100)
        self.v_head = nn.Linear(100, 1)

    def forward(self, x):
        x = F.relu(self.fc(x))
        state_value = self.v_head(x)
        return state_value


class Agent():

    clip_param = 0.1
    max_grad_norm = 0.3
    ppo_epoch = 10
    buffer_capacity, batch_size = 500, 32

    def __init__(self):
        self.training_step = 0
        self.anet = ActorNet().float()
        self.cnet = CriticNet().float()
        self.buffer = []
        self.counter = 0

        self.optimizer_a = optim.Adam(self.anet.parameters(), lr=1e-4)
        self.optimizer_c = optim.Adam(self.cnet.parameters(), lr=3e-4)

    def select_action(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        with torch.no_grad():
            action_probs = self.anet(state)
        dist = Categorical(action_probs)
        action = dist.sample()
        action_log_prob = dist.log_prob(action)
        return action.item(), action_log_prob.item()

    def get_value(self, state):

        state = torch.from_numpy(state).float().unsqueeze(0)
        with torch.no_grad():
            state_value = self.cnet(state)
        return state_value.item()

    def save_param(self):
        torch.save(self.anet.state_dict(), 'param/ppo_anet_params.pkl')
        torch.save(self.cnet.state_dict(), 'param/ppo_cnet_params.pkl')

    def store(self, transition):
        self.buffer.append(transition)
        self.counter += 1
        return self.counter % self.buffer_capacity == 0

    def update(self):
        self.training_step += 1

        s = torch.tensor([t.s for t in self.buffer], dtype=torch.float)
        a = torch.tensor([t.a for t in self.buffer], dtype=torch.float).view(-1, 1)
        r = torch.tensor([t.r for t in self.buffer], dtype=torch.float).view(-1, 1)
        s_ = torch.tensor([t.s_ for t in self.buffer], dtype=torch.float)

        old_action_log_probs = torch.tensor(
            [t.a_log_p for t in self.buffer], dtype=torch.float).view(-1, 1)

        # r = (r - r.mean()) / (r.std() + 1e-5)
        with torch.no_grad():
            target_v = r + args.gamma * self.cnet(s_)

        adv = (target_v - self.cnet(s)).detach()

        
        action_probs  = self.anet(s)
        dist = Categorical(action_probs)
        action_log_probs = dist.log_prob(a)
        ratio = torch.exp(action_log_probs - old_action_log_probs)
        # ratio = torch.exp(action_log_probs - old_action_log_probs[index].detach())

        surr1 = ratio * adv
        surr2 = torch.clamp(ratio, 1.0 - self.clip_param,
                            1.0 + self.clip_param) * adv
        action_loss = -torch.min(surr1, surr2).mean()

        self.optimizer_a.zero_grad()
        action_loss.backward()
        nn.utils.clip_grad_norm_(self.anet.parameters(), self.max_grad_norm)
        self.optimizer_a.step()

        value_loss = F.smooth_l1_loss(self.cnet(s), target_v)
        self.optimizer_c.zero_grad()
        value_loss.backward()
        nn.utils.clip_grad_norm_(self.cnet.parameters(), self.max_grad_norm)
        self.optimizer_c.step()

        del self.buffer[:]

EPISODE = 3000  # total training episodes
STEP = 5000  # step limitation in an episode
EVAL_EVERY = 10  # evaluation interval
TEST_NUM = 5  # number of tests every evaluation

def main():
    # env = gym.make('CartPole-v1', render_mode="human")
    env = gym.make('CartPole-v1')
    # env.seed(args.seed)
    num_state = env.observation_space.shape[0]
    num_action = env.action_space.n
    agent = Agent()

    training_records = []
    running_reward = -1000

    for i_ep in range(EPISODE):
        state, info = env.reset()

        for t in range(STEP):
            action, action_log_prob = agent.select_action(state)
            state_, reward, done, truncated, info = env.step(action) 
            agent.store(Transition(state, action, action_log_prob, reward, state_))
            state = state_
            if args.render:
                env.render()
            if done:
                agent.update()
                break

        if i_ep % args.log_interval == 0 or i_ep >= EPISODE - 1:
            total_reward = 0
            for i in range(TEST_NUM):
                state, info = env.reset()
                for _ in range(STEP):
                    action, _ = agent.select_action(state)
                    state_, reward, done, truncated, info = env.step(action)
                    total_reward += reward
                    state = state_
                    if done or truncated:
                        break
            score = total_reward / TEST_NUM
            print('Ep {}\ttest score: {:.2f}\t'.format(i_ep, score))
            training_records.append(TrainingRecord(i_ep, score))
        if score > 470:
            print("Solved! Moving average score is now {}!".format(running_reward))
            env.close()
            agent.save_param()
            with open('log/ppo_training_records.pkl', 'wb') as f:
                pickle.dump(training_records, f)
            break

    plt.plot([r.ep for r in training_records], [r.reward for r in training_records])
    plt.title('PPO')
    plt.xlabel('Episode')
    plt.ylabel('Moving averaged episode reward')
    plt.savefig("img/ppo.png")
    plt.show()


if __name__ == '__main__':
    main()
