#coding: utf-8
#date: 2016-05-30
#mail: artorius.mailbox@qq.com
#author: xinwangzhong -version 1.0

class HMMViterbi():
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
        
    def viterbi(self, observed):
        sta = []
        len_observed = len(observed)
        alpha = [([0]*self.len_hidden) for i in range(len_observed)]
        path = [([0]*self.len_hidden) for i in range(len_observed)]
        #第一天计算，状态的初始概率*隐藏状态到观察状态的条件概率
        for j in range(self.len_hidden):
            alpha[0][j]=self.pi[j]*self.B[j][self.observation.index(observed[0])]
            path[0][j] = -1
        # 第一天以后的计算
        # 前一天的每个状态转移到当前状态的概率最大值
        # *
        # 隐藏状态到观察状态的条件概率
        for i in range(1,len_observed):
            for j in range(self.len_hidden):
                max_ = 0.0
                index = 0
                for k in range(self.len_hidden):
                    if(alpha[i-1][k]*self.A[k][j] > max_):
                        max_ = alpha[i-1][k]*self.A[k][j]
                        index = k
                alpha[i][j] = max_ * self.B[j][self.observation.index(observed[i])]
                path[i][j] = index
        #找到最后一天天气呈现哪种观察状态的概率最大
        max_ = 0.0
        idx = 0
        for i in range(self.len_hidden):
            if(alpha[len_observed-1][i]>max_):
                max_ = alpha[len_observed-1][i]
                idx = i
        print "最可能隐藏序列的概率："+str(max_)
        sta.append(self.hidden[idx])
        #逆推回去找到每天出现哪个隐藏状态的概率最大
        for i in range(len_observed-1,0,-1):
            idx = path[i][idx]
            sta.append(self.hidden[idx])
        sta.reverse()
        return sta

if __name__ == '__main__':
    #假设观察到一组序列为observed，输出HMM模型（len_hidden，len_obs，A，B，pi）产生观察序列observed的概率
    observed = ['dry','damp','soggy']
    hmm_viterbi = HMMViterbi()
    print hmm_viterbi.viterbi(observed)
    
    # 最可能隐藏序列的概率：0.005184
    # ['rain', 'rain', 'sun']
    # [Finished in 0.2s]
