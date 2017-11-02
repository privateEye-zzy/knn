# K近邻算法，监督学习-分类算法 (你，就是你最常接触的五个人的平均)
import csv, numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
class KNearestNeighbor(object):
	def __init__(self):
		self.Run(split=0.75, k=10)
	# 绘制可视化图形之前的数据处理分类
	def dataForDraw(self):
		apple, orange, banana = [], [], []
		for line in self.dataset:
			if line[-1] == 'apple':
				apple.append(line[0:-1])
			elif line[-1] == 'orange':
				orange.append(line[0:-1])
			elif line[-1] == 'banana':
				banana.append(line[0:-1])
		return np.array(apple, dtype=np.float),np.array(orange, dtype=np.float), np.array(banana, dtype=np.float)
	# 可视化绘制2维数据
	def draw2D(self, x, y):
		apple, orange, banana = self.dataForDraw()
		plt.scatter(apple[:, x].tolist(), apple[:, y].tolist(), c='#FF4040')
		plt.scatter(orange[:, x].tolist(), orange[:, y].tolist(), c='#FF7F00')
		plt.scatter(banana[:, x].tolist(), banana[:, y].tolist(), c='#FFD700')
		plt.xlabel('X')
		plt.ylabel('Y')
		plt.show()
	# 可视化绘制3维数据
	def draw3D(self, x, y, z):
		ax = plt.figure().add_subplot(111, projection='3d')
		apple, orange, banana = self.dataForDraw()
		ax.scatter(apple[:, x].tolist(), apple[:, y].tolist(), apple[:, z].tolist(), c='#FF4040')
		ax.scatter(orange[:, x].tolist(), orange[:, y].tolist(), orange[:, z].tolist(), c='#FF7F00')
		ax.scatter(banana[:, x].tolist(), banana[:, y].tolist(), banana[:, z].tolist(), c='#FFD700')
		ax.set_zlabel('Z')
		ax.set_ylabel('Y')
		ax.set_xlabel('X')
		plt.show()
	# 样本集数据归一化
	def toOneDaDataset(self):
		temp = np.array(np.array(self.dataset)[:, 0:-1], dtype=np.float)
		# temp = ((temp - temp.mean()) / temp.std()).tolist()  # Z-score标准化
		temp = ((temp - temp.min()) / (temp.max() - temp.min())).tolist()  # min-max标准化
		[line.append(self.dataset[idx][-1]) for idx, line in enumerate(temp)]
		return temp
	# 导入解析数据
	def loadDataset(self, filename, split):
		with open(filename, 'r') as csvfile:
			self.dataset = list(csv.reader(csvfile))
			np.random.shuffle(self.dataset)
			self.dataset = self.toOneDaDataset()
			datasetSize = len(self.dataset)
			trainSet = self.dataset[0:int(split * datasetSize)]
			testSet = self.dataset[int(split * datasetSize):]
			return trainSet, testSet
	# 计算两个向量的欧式距离
	def calculateDistance(self, testInstance, trainInstance):
		test, train = np.array(testInstance[0:-1], dtype=np.float), np.array(trainInstance[0:-1], dtype=np.float)
		return np.sqrt(np.sum((test - train)**2))
	# 根据距离向量组，找到k个最近的邻居
	def getNeighbors(self, trainSet, testInstance, k):
		distances = []
		# 对训练集的每一个向量计算其到测试集的欧式距离
		for x in range(len(trainSet)):
			dist = self.calculateDistance(testInstance, trainSet[x])
			distances.append((trainSet[x], dist))
		distances = sorted(distances, key=lambda d: d[1])  # 距离排序
		neighbors = []
		[neighbors.append(distances[x][0]) for x in range(k)]  # 选取k个邻居
		return neighbors
	# k个邻居推断测试样本的类别
	def getLabel(self, neighbors):
		classVotes = {}
		for x in range(len(neighbors)):
			label = neighbors[x][-1]
			if label in classVotes:
				classVotes[label] += 1
			else:
				classVotes[label] = 1
		# 以少数服从多数倒叙排序Label
		sortedVotes = sorted(classVotes.items(), key=lambda cls: cls[1], reverse=True)
		return sortedVotes[0][0]  # 选取类别频率最大的的作为测试样本的类别
	# 算法准确率计算
	def getAccuracy(self, testSet, labels):
		correct = 0
		for x in range(len(testSet)):
			if testSet[x][-1] == labels[x]:
				correct += 1
		print('在{0}个测试数据中，共有{1}个标签预测正确'.format(len(testSet), correct))
		print('算法准确度为{0}%'.format(round((correct / float(len(testSet))) * 100.0, 2)))
	# 执行算法步骤
	def Run(self, split, k):
		trainSet, testSet = self.loadDataset('data/testKNN.csv', split)
		labels = []
		for x in range(len(testSet)):
			neighbors = self.getNeighbors(trainSet, testSet[x], k)
			labels.append(self.getLabel(neighbors))
		self.getAccuracy(testSet, labels)
		# self.draw3D(1, 2, 3)
		# self.draw2D(0, 3)
if __name__ == '__main__':
	knn = KNearestNeighbor()
