import logistic
dataArr,labelMat = logistic.loadDataSet()
print(dataArr,labelMat)
logistic.gradAscent(dataArr,labelMat)
print(logistic.gradAscent(dataArr,labelMat))