# Build a random Forest for the 'iris' data.

# Loading 'iris' dataset
iris<-read.csv("C:\\Users\\IN102385\\OneDrive - Super-Max Personal Care Pvt. Ltd\\Assignment - Data Science -UL\\Decision Tree-R\\iris.csv")
View(iris)
str(iris)
summary(iris)
# Declaring Species as a categorical variable(factor)
iris$Species<-as.factor(iris$Species)
str(iris)
# Loading required packages for EDA
install.packages("psych")
library(psych)
hist(iris$Sepal.Length)
hist(iris$Sepal.Width)
hist(iris$Petal.Length)
hist(iris$Petal.Width)
describe(iris)
boxplot(iris)
# Sepal Length and Sepal width follow normal distribution, Mean and Median are at centres
# Petal length and Petal width do not follow normal distribution; Means do not match with medians
# There are outliers in Sepal Width
pairs(iris)
# Based on the scatter diagram, relations are observed among sepal length,petal length and peta width
# Checking with Correlation coefficient values
cor(iris$Petal.Length,iris$Petal.Width)
cor(iris$Sepal.Length,iris$Petal.Length)
cor(iris$Sepal.Length,iris$Petal.Width)
cor(iris$Sepal.Width,iris$Petal.Length)
cor(iris$Sepal.Width,iris$Petal.Width)
# There is a strong positive relation among petal length,petal width and sepal length.
# Petal width does not have any strong relationship with any other variables

# Splitting dataset into training and testing
set.seed(7)
ind <- sample(2,nrow(iris), replace=TRUE, prob=c(0.7,0.3))
trainData <- iris[ind==1,]
testData <- iris[ind==2,]

# Application of RandomForest
# Loading the package
install.packages("randomForest")
library(randomForest)
#Generate Random Forest learning treee 
model <- randomForest(Species~.,data=trainData,ntree=100,proximity=TRUE)
# Evaluation of model performance
a<-table(predict(model),trainData$Species)
sum(diag(a)/sum(a))
# Training accuracy is 0.98

# Check the Random Forest model and importance features
print(model)
importance(model)
varImpPlot(model)
# The important features are observed to be Petal length and petal width

# prediction for testing data
irisPred<-predict(model,newdata=testData)
table(irisPred, testData$Species)
confusionMatrix(irisPred,testData$Species)
# Accuracy of the model is 0.9

# To check the margin, positive or negative, if positive it means correct classification
tune.rf <- tuneRF(iris[,-5],iris[,5], stepFactor=0.5)
print(tune.rf)
# With Mtry=1, we get minimum OOB error as 0.04
model2 <- randomForest(Species~., data=trainData, ntree = 140, mtry = 1, importance = TRUE,
                    proximity = TRUE)
a<-table(predict(model2),trainData$Species)
sum(diag(a)/sum(a))
# Training accuracy is 0.98
print(model2)
importance(model2)
varImpPlot(model2)
plot(model2)
# Again Petal Length and Petal Width are main contributors.
# Evaluation of 2nd model
irisPred2<-predict(model2,newdata=testData)
table(irisPred2, testData$Species)
confusionMatrix(irisPred2,testData$Species)
# Accuracy of the model is found to be 0.9 which is same as earlier
# We can plot the histogram with no.of nodes
hist(treesize(model2), main = "No of Nodes for the trees", col = "green")
# Majority of the trees has an average number has close to 35 to 40 nodes. 
partialPlot(model2, trainData, Petal.Length, "versicolor")
partialPlot(model2, trainData, Petal.Length, "setosa")
partialPlot(model2, trainData, Petal.Length, "virginica")
# if the petal.length is between 2.5 to 5,5, then it is Versicolor
# If the petal.length is between 1 to 3 cms in length, then it is setosa
# if the petal.length is greater than 3 cms in lenth, then it is Virginica

# Extract single tree from the forest :
tree <- getTree(model2, 1, labelVar = TRUE)
tree
# Multi Dimension scaling plot of proximity Matrix
MDSplot(model, iris$Species)
# Proximity plot indicates 3 different clusters for species

# CONCLUSION :
# The random forest algorithms accuracy of 0.9 for Iris data set
# Both the models also indicate that Petal length and petal width are important variables