#coding: utf-8
#date: 2016-06-05
#mail: artorius.mailbox@qq.com
#author: xinwangzhong -version 0.1

from math import log

class GoodTuring():
	def __init__(self, train_dict, test_dict):
		self.train_dict = train_dict
		self.test_dict = test_dict
		self.freq = {0:0}
		self.new_test_dict = {}

	def getFreq(self):
		# 计算在训练集中的词有多少个在测试集出现过c次
		for key in self.test_dict:
			if key in self.train_dict:
				if self.train_dict[key] not in self.freq:
					self.freq[self.train_dict[key]] = 0
				self.freq[self.train_dict[key]] += 1
			else:
				self.freq[0] += 1
		# 计算平滑后的c*
		for key in self.freq:
			if key+1 in self.freq:
				freq_tmp = 1.0*(key+1)*self.freq[key+1]/self.freq[key]
			else:
				freq_tmp = self.freq[key]
			self.freq[key] = freq_tmp
		# print self.freq
		for key in self.test_dict:
			if key in self.train_dict:
				self.new_test_dict[key] = self.freq[self.train_dict[key]]
			else:
				self.new_test_dict[key] = self.freq[0]
		return self.new_test_dict

if __name__ == '__main__':
	train_dict = {
	"what":2,
	"is":2,
	"s":1,
	"it":1,
	"small":1,
	"?":1,
	}
	test_dict = {
	"what":1,
	"is":1,
	"it":1,
	"small":1,
	"?":1,
	"s":1,
	"flying":1,
	"birds":1,
	"are":1,
	"bird":1,
	"a":1,
	".":1,
	}
	good_turing = GoodTuring(train_dict, test_dict)
	print good_turing.getFreq()
