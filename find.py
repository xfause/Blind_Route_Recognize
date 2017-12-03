import cv2
import numpy as np
import time

class Stack(object):
	def __init__(self):
		self.items = []
	def push(self, item):
		self.items.append(item)
	def pop(self):
		return self.items.pop()
	def peek(self):
		return self.items[-1]
	def isEmpty(self):
		return len(self.items) == 0

class Node(object):
	def __init__(self,index,x,y):
		self.index=index;
		self.x=x;
		self.y=y;


def regionGrow(task,height,weight):
	ClassNum =0
	stack1 = Stack();
	stack2 = Stack();
	for i in range(1,height):
		for j in range(1,weight):
			if (task[i][j][1]==0):
				task[i][j]=0
			elif (task[i][j][1]==255):
				stack1.push(i);
				stack2.push(j);
				index = 1
				ClassNum = ClassNum+1
				while(index>0):
					if (stack1.isEmpty() or stack2.isEmpty()):
						continue;
					x0 = stack1.pop()
					y0 = stack2.pop()
					index = index-1
					task[x0][y0]=ClassNum
					#print(x0,y0,ClassNum)
					# x0 y0 位于四个角
					if (x0==1 and y0==1):
						if (task[x0][y0+1][1]==255):
							task[x0][y0+1]=ClassNum
							index = index+1
							stack1.push(x0)
							stack2.push(y0+1)
						if (task[x0+1][y0+1][1]==255):
							task[x0+1][y0+1]=ClassNum
							index = index+1
							stack1.push(x0+1)
							stack2.push(y0+1)
						if (task[x0+1][y0][1]==255):
							task[x0+1][y0]=ClassNum
							index = index+1
							stack1.push(x0+1)
							stack2.push(y0)
					# 边缘
					elif (x0==1 and y0!=1 and y0!=weight):
						if (task[x0][y0+1][1]==255): # 0 +1
							task[x0][y0+1]=ClassNum
							index = index+1
							stack1.push(x0)
							stack2.push(y0+1)
						if (task[x0+1][y0+1][1]==255): # +1 +1
							task[x0+1][y0+1]=ClassNum
							index = index+1
							stack1.push(x0+1)
							stack2.push(y0+1)
						if (task[x0+1][y0][1]==255): # +1 0
							task[x0+1][y0]=ClassNum
							index = index+1
							stack1.push(x0+1)
							stack2.push(y0)
						if (task[x0][y0-1][1]==255): # 0 -1
							task[x0][y0-1]=ClassNum
							index = index+1
							stack1.push(x0)
							stack2.push(y0-1)
						if (task[x0+1][y0-1][1]==255): # +1 -1
							task[x0+1][y0-1]=ClassNum
							index = index+1
							stack1.push(x0+1)
							stack2.push(y0-1)
					# 边缘
					elif (y0==1 and x0!=1 and x0!=height):
						if (task[x0][y0+1][1]==255): # 0 +1
							task[x0][y0+1]=ClassNum;
							index = index+1
							stack1.push(x0)
							stack2.push(y0+1)
						if (task[x0+1][y0+1][1]==255): # +1 +1
							task[x0+1][y0+1]=ClassNum;
							index = index+1
							stack1.push(x0+1)
							stack2.push(y0+1)
						if (task[x0+1][y0][1]==255): # +1 0
							task[x0+1][y0]=ClassNum;
							index = index+1
							stack1.push(x0+1)
							stack2.push(y0)
						if (task[x0-1][y0+1][1]==255): # -1 +1
							task[x0-1][y0+1]=ClassNum;
							index = index+1
							stack1.push(x0-1)
							stack2.push(y0+1)
						if (task[x0-1][y0][1]==255): # -1 0
							task[x0-1][y0]=ClassNum;
							index = index+1
							stack1.push(x0-1)
							stack2.push(y0)
					elif (y0==weight and x0!=1 and x0!=height):
						if (task[x0+1][y0][1]==255): # +1 0
							task[x0+1][y0]=ClassNum
							index = index+1
							stack1.push(x0+1)
							stack2.push(y0)
						if (task[x0-1][y0][1]==255): # -1 0
							task[x0-1][y0]=ClassNum
							index = index+1
							stack1.push(x0-1)
							stack2.push(y0)
						if (task[x0-1][y0-1][1]==255): # -1 -1
							task[x0-1][y0-1]=ClassNum
							index = index+1
							stack1.push(x0-1)
							stack2.push(y0-1)
						if (task[x0+1][y0-1][1]==255): # +1 -1
							task[x0+1][y0-1]=ClassNum
							index = index+1
							stack1.push(x0+1)
							stack2.push(y0-1)
						if (task[x0][y0-1][1]==255): # 0 -1
							task[x0][y0-1]=ClassNum
							index = index+1
							stack1.push(x0)
							stack2.push(y0-1)
					elif (x0==height and y0!=1 and y0!=weight):
						if (task[x0-1][y0+1][1]==255): # -1 +1
							task[x0-1][y0+1]=ClassNum
							index = index+1
							stack1.push(x0-1)
							stack2.push(y0+1)
						if (task[x0-1][y0][1]==255): # -1 0
							task[x0-1][y0]=ClassNum
							index = index+1
							stack1.push(x0-1)
							stack2.push(y0)
						if (task[x0-1][y0-1][1]==255): # -1 -1
							task[x0-1][y0-1]=ClassNum
							index = index+1
							stack1.push(x0-1)
							stack2.push(y0-1)
						if (task[x0][y0-1][1]==255): # 0 -1
							task[x0][y0-1]=ClassNum
							index = index+1
							stack1.push(x0)
							stack2.push(y0-1)
						if (task[x0][y0+1][1]==255): # 0 +1
							task[x0][y0+1]=ClassNum
							index = index+1
							stack1.push(x0)
							stack2.push(y0+1)
					elif (x0==1 and y0==weight):
						if (task[x0][y0-1][1]==255): # 0 -1
							task[x0][y0-1]=ClassNum
							index = index+1
							stack1.push(x0)
							stack2.push(y0-1)
						if (task[x0+1][y0-1][1]==255): # +1 -1
							task[x0+1][y0-1]=ClassNum
							index = index+1
							stack1.push(x0+1)
							stack2.push(y0-1)
						if (task[x0+1][y0][1]==255): # +1 0
							task[x0+1][y0]=ClassNum
							index = index+1
							stack1.push(x0+1)
							stack2.push(y0)
					elif (x0==height and y0==1):
						if (task[x0-1][y0][1]==255): # -1 0
							task[x0-1][y0]=ClassNum
							index = index+1
							stack1.push(x0-1)
							stack2.push(y0)
						if (task[x0-1][y0+1][1]==255): # -1 +1
							task[x0-1][y0+1]=ClassNum
							index = index+1
							stack1.push(x0-1)
							stack2.push(y0+1)
						if (task[x0][y0+1][1]==255): # 0 +1
							task[x0][y0+1]=ClassNum
							index = index+1
							stack1.push(x0)
							stack2.push(y0+1)
					elif (x0==height and y0==weight):
						if (task[x0-1][y0][1]==255): # -1 0
							task[x0-1][y0]=ClassNum
							index = index+1
							stack1.push(x0-1)
							stack2.push(y0)
						if (task[x0-1][y0-1][1]==255): # -1 -1
							task[x0-1][y0-1]=ClassNum
							index = index+1
							stack1.push(x0-1)
							stack2.push(y0-1)
						if (task[x0][y0-1][1]==255): # 0 -1
							task[x0][y0-1]=ClassNum
							index = index+1
							stack1.push(x0)
							stack2.push(y0-1)
					else:
						if (task[x0][y0+1][1]==255): # 0 +1
							task[x0][y0+1]=ClassNum
							index = index+1
							stack1.push(x0)
							stack2.push(y0+1)
						if (task[x0+1][y0+1][1]==255): # +1 +1
							task[x0+1][y0+1]=ClassNum
							index = index+1
							stack1.push(x0+1)
							stack2.push(y0+1)
						if (task[x0+1][y0][1]==255): # +1 0
							task[x0+1][y0]=ClassNum
							index = index+1
							stack1.push(x0+1)
							stack2.push(y0)
						if (task[x0][y0-1][1]==255): # 0 -1
							task[x0][y0-1]=ClassNum
							index = index+1
							stack1.push(x0)
							stack2.push(y0-1)
						if (task[x0+1][y0-1][1]==255): # +1 -1
							task[x0+1][y0-1]=ClassNum
							index = index+1
							stack1.push(x0+1)
							stack2.push(y0-1)
						if (task[x0-1][y0+1][1]==255): # -1 +1
							task[x0-1][y0+1]=ClassNum
							index = index+1
							stack1.push(x0-1)
							stack2.push(y0+1)
						if (task[x0-1][y0][1]==255): # -1 0
							task[x0-1][y0]=ClassNum
							index = index+1
							stack1.push(x0-1)
							stack2.push(y0)
						if (task[x0-1][y0-1][1]==255): # -1 -1
							task[x0-1][y0-1]=ClassNum
							index = index+1
							stack1.push(x0-1)
							stack2.push(y0-1)
	#task[229][173]=[255,255,255]
	return task,ClassNum

def Review(task,Num,height,weight):
	for i in range(1,height):
		for j in range(1,weight):
			if (task[i][j][1]==Num):
				task[i][j]=[255,255,255]
			else:
				task[i][j]=[0,0,0]
	return task
	pass

def Count(task,ClassNum,height,weight):
	a=[0]*ClassNum
	for i in range(1,height):
		for j in range(1,weight):
			if (task[i][j].all()!=0):
				a[task[i][j][1]]=a[task[i][j][1]]+1
	for i in range(1,ClassNum):
		if (a[i]>6000):
			return i
			break;
	pass
if __name__=="__main__":
	img = cv2.imread("route8.jpg")

	# time start
	start = time.clock()
	
	hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV) #HSV空间
	sp = hsv.shape
	height = sp[0]
	weight = sp[1]
	#print(height,weight)
	H,S,V = cv2.split(hsv)

	task = hsv
	
	for i in range(0,height):
		for j in range(0,weight):
			if (H[i][j]>10 and H[i][j]<40 and S[i][j]>70 and S[i][j]<255 and V[i][j]>70 and V[i][j]<255):
				task[i][j]=255
			else:
				task[i][j]=0

	# 中值滤波 平滑
	task1 = cv2.medianBlur(task,11)
	# 膨胀
	square = cv2.getStructuringElement(cv2.MORPH_CROSS,(5,5)) 
	task2 = cv2.erode(task1,square)
	# 区域生长
	task3,ClassNum= regionGrow(task2,height-1,weight-1)
	# 统计
	Num = Count(task3,ClassNum+1,height-1,weight-1)
	task4 = Review(task3,Num,height-1,weight-1)
	
	# time over
	elapsed = (time.clock() - start)
	print("Time used:",elapsed)

	# cv2.imshow('img',img)
	cv2.imshow('result',task4)
	# cv2.imshow('res',res)
	cv2.waitKey(0)
	cv2.destroyAllWindows()