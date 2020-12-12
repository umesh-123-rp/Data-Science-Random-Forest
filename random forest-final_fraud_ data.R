###Problem Statement:
# Use Random Forest to prepare a model on fraud data 
# treating those who have taxable_income <= 30000 as "Risky" and others are "Good"

#Data Description :
  
#Undergrad : person is under graduated or not
#Marital.Status : marital status of a person
#Taxable.Income : Taxable income is the amount of how much tax an individual owes to the government 
#Work Experience : Work experience of an individual person
#Urban : Whether that person belongs to urban area or not
# Loading the required packages
install.packages("caret")
install.packages("gmodels")
install.packages("psych")
install.packages("randomForest")
library(randomForest)
library(caret)
library(gmodels)
library(psych)
# Loading the dataset 
Fraud_Data<-read.csv("C:\\Users\\IN102385\\OneDrive - Super-Max Personal Care Pvt. Ltd\\Assignment - Data Science -UL\\Decision Tree-R\\Fraud_check.csv")
View(Fraud_Data)
str(Fraud_Data)
# Understanding the distribution of taxable income
hist(Fraud_Data$Taxable.Income)
# Histogram with lower intervals for better understanding
hist(Fraud_Data$Taxable.Income, main = "Taxable.Income",xlim = c(0,100000),
     breaks=c(seq(40,60,80)), col = c("blue","red", "green","violet"))
plot(Fraud_Data$Taxable.Income)
summary(Fraud_Data)
describe(Fraud_Data)
# The taxable income is uniform throughout

# Defining the categorical data in the given data set
Fraud_Data$Undergrad<-as.factor(Fraud_Data$Undergrad)
Fraud_Data$Marital.Status<-as.factor(Fraud_Data$Marital.Status)
Fraud_Data$Urban<-as.factor(Fraud_Data$Urban)
pairs(Fraud_Data)
# From the scatter diagram, it indicates that there is no any relationship among the variables 

# Categorise Taxable Income in the form Risky and Good
Risky_Good = ifelse(Fraud_Data$Taxable.Income<=30000, "Risky", "Good")

# Adding Risky_Good as a column in the company dataset
FD = data.frame(Fraud_Data, Risky_Good)

# Declaring the Risky_Good as a categorical variable
FD$Risky_Good<-as.factor(FD$Risky_Good)
View(FD)
FD1<-FD[,-3]
table(FD1$Risky_Good)
# Splitting data into training and testing
inTraininglocal<-createDataPartition(FD1$Risky_Good,p=.70,list = F)
training<-FD1[inTraininglocal,]
testing<-FD1[-inTraininglocal,]

#Model Building using Random Forest algorithm
attach(FD1)
set.seed(213)
model<-randomForest(Risky_Good~., data=training)
print(model)
plot(model)
# OOB error was found to be 22%
# OOB error is getting stabilised after 20 nos. of trees

# To understand the importance of variables,reduction in Gini impurity was calculated
print(importance(model))
varImpPlot(model)
# Maximum decrease was found in City Population and Work Experience

# Evaluation of performance 
pred<-predict(model, training)
a<-table(training$Risky_Good,pred)
sum(diag(a))/sum(a)
# Training accuracy is 0.92

# Prediction with test data
pred1<-predict(model,testing[-6])
a<-table(testing$Risky_Good,pred1)
sum(diag(a))/sum(a)
# Testing accuracy is 0.79

# Further tuning the random forest model with no. of variables available for splitting 
tune <- tuneRF(FD[,-6], FD[,6], stepFactor = 0.5, plot = TRUE, ntreeTry = 300,
               trace = TRUE, improve = 0.05)
# We observe that mtry = 2 gives minimum OOB of 45%
attach(FD1)
# We can now optimise the model with mtry=2 as follows :
model_rf1 <- randomForest(Risky_Good~., data=training, ntree = 300, mtry = 2, importance = TRUE,
                          proximity = TRUE)
model_rf1

# Evaluation of model performance
pred_rf1 <- predict(model_rf1, training)
confusionMatrix(pred_rf1, training$Risky) 
# training accuracy is observed to be 0.92

# test data prediction using the Tuned RF1 model
pred_rf2 <- predict(model_rf1, testing[,-6])
confusionMatrix(pred_rf2, testing$Risky_Good) 
# Testing Accuracy is observed to be 0.79 

# Frequency diagram for no. of nodes
hist(treesize(model_rf1), main = "No of Nodes for the trees", col = "green")
# Majority of the trees has an average number of 70 to 80 nodes. 
varImpPlot(model_rf1)
# City Population and Work Experience are important variables

# Extract single tree from the forest :
getTree(model, 1, labelVar = TRUE)

# Multi-Dimension scaling plot of proximity Matrix
MDSplot(model_rf1, FD1$Risky_Good)

# CONCLUSION :
# Random Forest algorithm gives an accuracy of 0.79 against the dataset
# The same dataset can be used for building the models with help of other supervised machine learning algorithm.
