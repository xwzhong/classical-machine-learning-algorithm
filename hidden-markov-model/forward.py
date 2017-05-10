#coding: utf-8
#date: 2016-05-29
#mail: artorius.mailbox@qq.com
#author: xinwangzhong -version 1.0

class HMMForward():
    def __init__(self):
        # 3种隐藏层状态:sun cloud rain
        self.hidden = []
        self.hidden.append('sun')
        self.hidden.append('cloud')
        self.hidden.append('rain')
        self.len_hidden = len(self.hidden)
        # 3种观察层状态:dry damp soggy
        self.observation = []
        self.observation.append('dry')
        self.observation.append('damp')
        self.observation.append('soggy')
        self.len_obs = len(self.observation)
        # 初始状态矩阵（1*N第一天是sun，cloud，rain的概率）
        self.pi = (0.3,0.3,0.4)
        # 状态转移矩阵A（len_hidden*len_hidden 隐藏层状态之间互相转变的概率）
        self.A=((0.2,0.3,0.5),(0.1,0.5,0.4),(0.6,0.1,0.3))
        # 混淆矩阵B（len_hidden*len_obs 隐藏层状态对应的观察层状态的概率）
        self.B=((0.1,0.5,0.4),(0.2,0.4,0.4),(0.3,0.6,0.1))
        
    def forward(self, observed):
        p = 0.0
        #观察到的状态数目
        len_observed = len(observed)
        #中间概率 len_observed*len_obs
        alpha = [([0]*self.len_hidden) for i in range(len_observed)]
        #第一个观察到的状态,状态的初始概率乘上隐藏状态到观察状态的条件概率。
        for j in range(self.len_hidden):
            alpha[0][j] = self.pi[j]*self.B[j][self.observation.index(observed[0])]
        #第一个之后的状态，首先从前一天的每个状态，转移到当前状态的概率求和，然后乘上隐藏状态到观察状态的条件概率。
        for i in range(1, len_observed):
            for j in range(self.len_hidden):
                sum_tmp = 0.0
                for k in range(self.len_hidden):
                    sum_tmp += alpha[i-1][k]*self.A[k][j]
                alpha[i][j] = sum_tmp * self.B[j][self.observation.index(observed[i])]
        for i in range(self.len_hidden):
            p += alpha[len_observed-1][i]
        return p

if __name__ == '__main__':
    #假设观察到一组序列为observed，输出HMM模型（len_hidden，len_obs，A，B，pi）产生观察序列observed的概率
    observed = ['dry']
    hmm_forword = HMMForward()
    print hmm_forword.forward(observed)
    observed = ['damp']
    print hmm_forword.forward(observed)
    observed = ['dry','damp']
    print hmm_forword.forward(observed)
    observed = ['dry','damp','soggy']
    print hmm_forword.forward(observed)
    # 
    # 0.21
    # 0.51
    # 0.1074
    # 0.030162
    # [Finished in 0.2s]
