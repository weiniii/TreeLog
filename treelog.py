import numpy as np
import tqdm
import time
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--EPSILON', default=1, type=float, help='the parameter for differential privacy')
    parser.add_argument('--DELTA', default=0.99, type=float, help='the parameter for differential privacy')
    parser.add_argument('--DOMAIN_SIZE', default=2**16, type=int, help='the domain size')
    parser.add_argument('--N_SAMPLES', default=10 , type=int, help='the sample size')
    parser.add_argument('--t', default=0, type=int, help='the parameter in the TreeLog')
    args = parser.parse_args(args=[])
    return args

args = parse_args()

def log_star(nums):
    times = 0
    while nums > 1:
        nums = np.log2(nums)
        times += 1
    return times

def get_geometric_noise():
    return np.random.geometric(1 - np.exp(-args.EPSILON)) + 1

def get_laplace_noise():
    return np.random.laplace(1 / args.EPSILON)

def interior_points_function(D, z):    
    return min(np.sum(D <= z), np.sum(D >= z))

def exponential_mechanism(scores):
    probability = np.exp(args.EPSILON * scores / 2)
    probability /= np.sum(probability)
    idx = np.arange(len(scores))
    mechanism = np.random.choice(idx, 1, p=probability)

    return mechanism

def Constuct_Tree(D, DOMAIN_SIZE):
    ARRAY_LEN = DOMAIN_SIZE * 2
    TREE = np.zeros(ARRAY_LEN).astype(int)
    for idx in (D - 1):
        TREE[idx + DOMAIN_SIZE] += 1
    for idx in range(len(TREE) - 1, 1, -1):
        if TREE[idx] > 0:
            parent = int(idx / 2)
            TREE[parent] += TREE[idx]
        else:
            pass
    return TREE

def print_tree(TREE):
    for i in range(args.N + 1):
        print(TREE[2 ** i : 2 ** (i + 1)])
        
        
def left_most(cur_other, DOMAIN_SIZE):
    most_left = cur_other
    while most_left < DOMAIN_SIZE: most_left *= 2
    most_left = most_left % DOMAIN_SIZE 
    return most_left

def right_most(cur_other, DOMAIN_SIZE):
    most_right = cur_other
    while most_right < DOMAIN_SIZE: most_right = 1 + (most_right << 1)
    most_right = most_right %  DOMAIN_SIZE
    return most_right

def embed(D, DOMAIN_SIZE):
        
    TREE = Constuct_Tree(D, DOMAIN_SIZE)
    D_new = np.zeros(DOMAIN_SIZE).astype(int)
    q = 1
    GAMMA = 0
    cur = 1
    while cur < DOMAIN_SIZE:
        cur_left, cur_right = cur * 2, cur * 2 + 1
        w_cur_left, w_cur_right = TREE[cur_left], TREE[cur_right]
        GAMMA = max(GAMMA, min(w_cur_left, w_cur_right))
        if w_cur_left >= w_cur_right:
            cur_next, cur_other = cur_left, cur_right
        else:
            cur_next, cur_other = cur_right, cur_left
        
        most_left, most_right = left_most(cur_other, DOMAIN_SIZE), right_most(cur_other, DOMAIN_SIZE)
        
        D_new[most_left : most_right + 1] = q
        
        cur = cur_next
        q += 1

    D_new[np.where(D_new == 0)] = int(np.log2(DOMAIN_SIZE))
    D_new = np.concatenate((D_new[D - 1].reshape(-1, 1), D.reshape(-1, 1)), axis=1)
    D_new = D_new[np.lexsort(D_new[:,::-1].T)]
    
    return D_new, GAMMA

def OneHeavyRound(D, DOMAIN_SIZE):
    TREE = Constuct_Tree(D, DOMAIN_SIZE)
    cur = 1
    rho = get_laplace_noise()
    while cur < DOMAIN_SIZE:
        cur_left, cur_right = cur * 2, cur * 2 + 1
        w_cur_left, w_cur_right = TREE[cur_left], TREE[cur_right]
        w_min = min(w_cur_left, w_cur_right)
        
        if w_min > args.t / 10 and w_min + get_laplace_noise() >= args.t / 4 + rho:
            cur_left = cur * 2
            return right_most(cur_left, DOMAIN_SIZE) 
        
        if w_cur_left >= w_cur_right:
            cur_next, cur_other = cur_left, cur_right
        else:
            cur_next, cur_other = cur_right, cur_left
        
                
        cur = cur_next
    return cur

def SelectAndCompute(D, t, E, DOMAIN_SIZE=None):
    if type(DOMAIN_SIZE) == type(None):
        t = t + get_geometric_noise()
    else:
        t = 2 * t + get_geometric_noise()
        
    if E == '<':
        D = np.sort(D)
        S, D = D[:t], D[t:]
        return D, S
    elif E == '>':
        D = np.sort(D)
        S, D = D[-t:], D[:-t]
        return D, S
    elif E == 'embed':
        D_new, gamma = embed(D, DOMAIN_SIZE)
        S_d, D_new = D_new[:t], D_new[t:]
        return D_new, S_d, gamma


def TreeLog(X, D):
    DOMAIN_SIZE = len(X)
    
    if DOMAIN_SIZE <= 16:
        scores = np.array([interior_points_function(D, x) for x in X])
        return exponential_mechanism(scores)
    
    if np.log2(DOMAIN_SIZE) % 1:
        DOMAIN_SIZE = int(2 ** np.ceil(np.log2(DOMAIN_SIZE)))
        ddd = np.arange(1, DOMAIN_SIZE + 1)
    
    D_1, S_left = SelectAndCompute(D, args.t, '<')
    D_2, S_right = SelectAndCompute(D_1, args.t, '>')
    D_border = np.concatenate((D_1, D_2))
    D_new, S_d, gamma = SelectAndCompute(D_2, args.t, 'embed', DOMAIN_SIZE)
    
    if gamma + get_laplace_noise() >= 3 * args.t / 4 + get_laplace_noise():
        del D_new
        return True, OneHeavyRound(D, DOMAIN_SIZE)
    
    D_new, S_d = D_new[:, 0], S_d[:, 1]
    
    EMBED_DOMAIN_SIZE = np.log2(DOMAIN_SIZE).astype(int)
    Y = np.arange(1, EMBED_DOMAIN_SIZE + 1)
    y_hat = TreeLog(Y, D_new)

    TREE = Constuct_Tree(S_d, DOMAIN_SIZE)
    front_idx = 2 ** y_hat
    end_idx = 2 ** (y_hat + 1)
    scores = TREE[front_idx:end_idx]

    v_hat = exponential_mechanism(scores) + front_idx
    v_left_most = left_most(v_hat, DOMAIN_SIZE)
    v_left_right = right_most(v_hat * 2, DOMAIN_SIZE)
    v_right_most = right_most(v_hat, DOMAIN_SIZE)
    v_right_left = left_most(v_hat * 2 + 1, DOMAIN_SIZE)

    C = [v_left_most, v_right_most, v_left_right, v_right_left]
    scores = np.array([interior_points_function(D_border, x) for x in C])

    idx = exponential_mechanism(scores)
    output = D_border[idx]

    return False, output

def success(D, value):
    if np.min(D) <= value and value <= np.max(D):
        return True
    else:
        return False