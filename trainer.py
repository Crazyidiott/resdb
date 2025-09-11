import grpc
from concurrent import futures
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import monitor_pb2
import monitor_pb2_grpc
import os
import time
import threading

# PPO Hyperparameters
learning_rate = 0.0005
gamma = 0.98
lmbda = 0.95
eps_clip = 0.1
K_epoch = 3
T_horizon = 20

# Value arrays for actions
A_values = torch.arange(1000, 20001, 1000)
B_values = torch.arange(100, 5100, 100)
C_values = torch.tensor([8, 16, 32, 64, 128, 256, 512, 1024])

# Reward weights
c_thr = 1.0
c_lat = -100.0

# Globals
model_path = './model.pth'
scores = open("./score.txt", 'w')
old_state = []
score = 0

addr = ['172.31.34.221:45678',
        '172.31.41.91:45678',
        '172.31.41.128:45678',
        '172.31.39.209:45678']
f = int((len(addr)-1)/3)
QUORUM = 2 * f + 1
action_pool = []
enough_action = threading.Condition()
action_lock = threading.Lock()

notify_counter = 0
max_notify_count = 200000

lastA, lastB, lastC, lastO = 0, 0, 0, 0

class PPO(nn.Module):
    def __init__(self):
        super(PPO, self).__init__()
        self.data = []

        input_dims = 6
        hidden_dims = 64

        self.fc1 = nn.Linear(input_dims, hidden_dims)
        self.fc_pi_a = nn.Linear(hidden_dims, len(A_values))
        self.fc_pi_b = nn.Linear(hidden_dims, len(B_values))
        self.fc_pi_c = nn.Linear(hidden_dims, len(C_values))
        self.fc_v = nn.Linear(hidden_dims, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def pi(self, x, softmax_dim=-1):
        x = F.relu(self.fc1(x))
        prob_a = F.softmax(self.fc_pi_a(x), dim=softmax_dim)
        prob_b = F.softmax(self.fc_pi_b(x), dim=softmax_dim)
        prob_c = F.softmax(self.fc_pi_c(x), dim=softmax_dim)
        return prob_a, prob_b, prob_c

    def v(self, x):
        x = F.relu(self.fc1(x))
        return self.fc_v(x)

    def put_data(self, transition):
        self.data.append(transition)

    def make_batch(self):
        s_lst, a_lst, b_lst, c_lst, r_lst, s_prime_lst, prob_a_lst, prob_b_lst, prob_c_lst = [], [], [], [], [], [], [], [], []

        for transition in self.data:
            s, a, b, c, r, s_prime, p_a, p_b, p_c = transition
            s_lst.append(s)
            a_lst.append([a])
            b_lst.append([b])
            c_lst.append([c])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            prob_a_lst.append([p_a])
            prob_b_lst.append([p_b])
            prob_c_lst.append([p_c])

        s, a, b, c, r = torch.tensor(s_lst, dtype=torch.float), torch.tensor(a_lst), torch.tensor(b_lst), torch.tensor(c_lst), torch.tensor(r_lst)
        s_prime, prob_a, prob_b, prob_c = torch.tensor(s_prime_lst, dtype=torch.float), torch.tensor(prob_a_lst), torch.tensor(prob_b_lst), torch.tensor(prob_c_lst)
        self.data = []
        return s, a, b, c, r, s_prime, prob_a, prob_b, prob_c
        

    def train_net(self):
        s, a, b, c, r, s_prime, prob_a, prob_b, prob_c = self.make_batch()
        for i in range(K_epoch):
            td_target = r + gamma * self.v(s_prime)
            delta = td_target - self.v(s)
            delta = delta.detach().numpy()

            advantage_lst = []
            advantage = 0.0
            for delta_t in delta[::-1]:
                advantage = gamma * lmbda * advantage + delta_t[0]
                advantage_lst.append([advantage])
            advantage_lst.reverse()
            advantage = torch.tensor(advantage_lst, dtype=torch.float)

            mask_a, mask_b, mask_c = (a!=0).float(), (b!=0).float(), (c!=0).float()

            pi_a, pi_b, pi_c = self.pi(s, softmax_dim=1)
            pi_a, pi_b, pi_c = pi_a.gather(1, a), pi_b.gather(1, b), pi_c.gather(1, c)
            ratio_a = torch.exp(torch.log(pi_a) - torch.log(prob_a))
            ratio_b = torch.exp(torch.log(pi_b) - torch.log(prob_b))
            ratio_c = torch.exp(torch.log(pi_c) - torch.log(prob_c))

            surr1 = ratio_a*advantage*mask_a + ratio_b*advantage*mask_b + ratio_c*advantage*mask_c
            surr2 = torch.clamp(ratio_a, 1 - eps_clip, 1 + eps_clip) * advantage * mask_a + \
                    torch.clamp(ratio_b, 1 - eps_clip, 1 + eps_clip) * advantage * mask_b + \
                    torch.clamp(ratio_c, 1 - eps_clip, 1 + eps_clip) * advantage * mask_c
            loss = -torch.min(surr1, surr2).mean() + F.smooth_l1_loss(self.v(s), td_target.detach())

            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

    def select_action(self, s):
        prob_a, prob_b, prob_c = self.pi(torch.from_numpy(s).float())
        a = Categorical(prob_a).sample().item()
        b = Categorical(prob_b).sample().item()
        c = Categorical(prob_c).sample().item()
        return a, b, c

model = PPO()
if os.path.exists(model_path):
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state'])
    model.optimizer.load_state_dict(checkpoint['optimizer_state'])


def Normalized(state):
    return (state - np.mean(state, axis=0)) / np.std(state, axis=0)

def modelUpdate(batch_state, option):
    global model, scores, old_state, score, lastA, lastB, lastC, lastO

    if len(old_state) == 0:
        old_state = batch_state
        lastO = option
        if option == 1:
            lastA, lastB = 10000, 1000
            return 10000, 1000
        else :
            lastC = 128
            return 128, 0

    reward = c_thr * old_state[0] + c_lat * old_state[1]

    norm_state = Normalized(old_state)
    prob_a, prob_b, prob_c = model.pi(torch.from_numpy(norm_state).float())
    a = (torch.abs(A_values - lastA)).argmin().item();
    b = (torch.abs(B_values - lastB)).argmin().item();
    c = (torch.abs(C_values - lastC)).argmin().item();

    if lastO == 1:
        model.put_data((norm_state, a, b, 0, reward / 40000, Normalized(batch_state),
                        prob_a[a].item(), prob_b[b].item(), 1.0))
    else :
        model.put_data((norm_state, 0, 0, c, reward / 40000, Normalized(batch_state),
                        1.0, 1.0, prob_c[c].item()))
    score += reward / 40000

    if len(model.data) >= T_horizon:
        model.train_net()
        # torch.save({
        #     'model_state': model.state_dict(),
        #     'optimizer_state': model.optimizer.state_dict()
        # }, model_path)

    old_state, lastO = batch_state, option

    a, b, c = model.select_action(Normalized(old_state))
    if option == 1:
        lastA, lastB = A_values[a].item(), B_values[b].item()
        return lastA, lastB
    else :
        lastC = C_values[c].item()
        return lastC, 0


class MetricsServiceServicer(monitor_pb2_grpc.MetricsServiceServicer):
    def SendMetrics(self, request, context):
        global notify_counter, lastC, action_pool
        new_state = np.array([request.throughput, request.latency, request.size,
                              request.tot_thr, request.tot_lat, request.tot_size])
        action_pool.clear()

        notify_counter += 1
        if notify_counter > max_notify_count:
            print("One cycle completed. Quit running.")
            os._exit(0)

        start = time.perf_counter()

        primary, id = int(request.option / 100) % 100, int(request.option / 10000)
        batchSize, batchTimeout = modelUpdate(new_state, request.option % 100)

        end = time.perf_counter()
        scores.write(f"代码运行时间: {end - start:.6f} 秒")
        scores.flush()

        option = request.option % 100
        if option == 1:
            return monitor_pb2.MetricsResponse(BatchSize=batchSize, BatchTimeout=batchTimeout)
        else:
            scores.write(f'{batchSize}\n')
            scores.flush()
            time.sleep(1)
            if id != primary:
                with grpc.insecure_channel(addr[primary-1]) as channel:
                    stub = monitor_pb2_grpc.MetricsServiceStub(channel)
                    response = stub.Connect(monitor_pb2.Timestamp(timestamp=batchSize))
                    SeqNum = response.timestamp
            else :
                response = self.Connect(monitor_pb2.Timestamp(timestamp=batchSize), None)
                SeqNum = response.timestamp
            lastC = SeqNum
            scores.write(f'{SeqNum}\n')
            scores.flush()
            return monitor_pb2.MetricsResponse(BatchSize=SeqNum, BatchTimeout=0)
        
    def Connect(self, request, context):
        recv_value = request.timestamp
        scores.write(f'{recv_value}\n')
        scores.flush()
        scores.write(str(action_pool) + '\n')
        scores.flush()
        with enough_action:
            with action_lock:
                action_pool.append(recv_value)

            if len(action_pool) >= QUORUM:
                enough_action.notify_all()

            while len(action_pool) < QUORUM:
                enough_action.wait()

            result = int(np.median(action_pool))
        
        return monitor_pb2.Timestamp(timestamp=result)



def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    monitor_pb2_grpc.add_MetricsServiceServicer_to_server(MetricsServiceServicer(), server)
    server.add_insecure_port('[::]:45678')
    server.start()
    print("Server started on port 45678.")
    server.wait_for_termination()


if __name__ == '__main__':
    serve()