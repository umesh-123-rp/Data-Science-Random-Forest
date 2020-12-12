###Problem Statement:
#A cloth manufacturing company is interested to know about the segment or attributes causes high sale. 
#Approach - A Random Forest can be built with target variable Sales 
#(we will first convert it in categorical variable) & all other variable 
#will be independent in the analysis.  

#About the data: 
#Let's consider a Company dataset with around 10 variables and 400 records. 
#The attributes are as follows: 
#Sales -- Unit sales (in thousands) at each location
#Competitor Price -- Price charged by competitor at each location
#Income -- Community income level (in thousands of dollars)
#Advertising -- Local advertising budget for company at each location (in thousands of dollars)
#Population -- Population size in region (in thousands)
#Price -- Price company charges for car seats at each site
#Shelf Location at stores -- A factor with levels Bad, Good and Medium indicating the quality of the shelving location for the car seats at each site
#Age -- Average age of the local population
#Education -- Education level at each location
#Urban -- A factor with levels No and Yes to indicate whether the store is in an urban or rural location
#US -- A factor with levels No and Yes to indicate whether the store is in the US or not
# Loading the required packages
install.packages("caret")
install.packages("gmodels")
install.packages("psych")
library(caret)
library(gmodels)
library(psych)
# Loading the dataset 
Company_Data<-read.csv("C:\\Users\\IN102385\\OneDrive - Super-Max Personal Care Pvt. Ltd\\Assignment - Data Science -UL\\Decision Tree-R\\Company_Data.csv")
View(Company_Data)
str(Company_Data)
# Understanding the distribution of Sales
hist(Company_Data$Sales, main = "Sales of Companydata",xlim = c(0,20),
     breaks=c(seq(10,20,30)), col = c("blue","red", "green","violet"))
plot(Company_Data$Sales)
summary(Company_Data)
describe(Company_Data)
# The sales distribution seems to be normally distributed

# Defining the categorical data in the given data set
Company_Data$ShelveLoc<-as.factor(Company_Data$ShelveLoc)
Company_Data$Urban<-as.factor(Company_Data$Urban)
Company_Data$US<-as.factor(Company_Data$US)
pairs(Company_Data)
# From the scatter diagram, it indicates that there is a decreasing relationship between price and Sales
# And also there is an increasing relatioship between competitor price and Price
# To check the strength of relation, let's calculate correlation coefficient
cor(Company_Data$Sales,Company_Data$Price)
cor(Company_Data$CompPrice,Company_Data$Price)
# Correlation coefficient is -0.44 for price and sales
# Correlation coefficient is 0.58 between Competitor price and price
# The relation is not very strong

# We can start building a linear regression with the dataset
attach(Company_Data)

###Model Building process by applying Random Forest Algorithm#######

# Defining Sales in the form of categorical variable
High = ifelse(Company_Data$Sales<9, "No", "Yes")
# Adding High as a column in the company dataset
CD = data.frame(Company_Data, High)
View(CD)
table(CD$High)
# Remove the column Sales which got replaced by the column "High"
CD1<-CD[,-1]
# Declaring the column "High" as a factor
CD1$High<-as.factor(CD1$High)
View(CD1)
# Splitting data into training and testing
inTraininglocal<-createDataPartition(CD1$High,p=.70,list = F)
training<-CD1[inTraininglocal,]
testing<-CD1[-inTraininglocal,]

# To check further improvement, random forest algorithm was applied
# Random Forest
install.packages("randomForest")
library(randomForest)
model_rf<-randomForest(High~., data=training)
print(model_rf)
plot(model_rf)
# With 500 no.f trees, OOB = 14.95%

# To understand the importance of variables,reduction in Gini impurity was calculated
print(importance(model_rf))
# Maximum decrease was found in Price and then ShelveLoc

# Evaluation of performance
pred_rf<-predict(model_rf,testing[,-11])
a<-table(testing$High,pred_rf)
sum(diag(a))/sum(a)
# Testing accuracy was found to be 0.82.

# Further tuning the random forest model with no. of variables available for splitting 
tune <- tuneRF(CD1[,-11], CD1[,11], stepFactor = 0.5, plot = TRUE, ntreeTry = 300,
               trace = TRUE, improve = 0.05)
# We observe that mtry = 3 gives minimum OOB of 13%

# We can now optimise the model with mtry=3 as follows :
model_rf1 <- randomForest(High~., data=training, ntree = 300, mtry = 3, importance = TRUE,
                    proximity = TRUE)
model_rf1

# Evaluation of model performance
pred_rf1 <- predict(model_rf1, training)
confusionMatrix(pred_rf1, training$High) 
# 100 % accuracy on training data 

# Accuracy is0.98 to1 at 95% Confidence Interval. 
# Sensitivity for Yes and No is 100 % 

# test data prediction using the Tuned RF1 model
pred_rf2 <- predict(model_rf1, testing)
confusionMatrix(pred_rf2, testing$High) 
# Testing Accuracy is observed to be 0.84 

# Plotting histogram of no. of nodes
hist(treesize(model_rf1), main = "No of Nodes for the trees", col = "green")
# Majority of the trees has an average number of 45 to 50 nodes. 

# The important variables can be observed
varImpPlot(model_rf1)
# The important variables are Price and SheleLoc

# Top 5 important variables are as follows :
varImpPlot(model_rf1 ,Sort = T, n.var = 5, main = "Top 5 -Variable Importance")
# The important variables in sequence are price, ShelveLoc, Age, Advertising and CompPrice

partialPlot(model_rf1, training, Price, "Yes")
# On that graph, i see that if the price is 100 or greater,
# then there is reduction in buying

# Extract single tree from the forest :
getTree(model_rf, 1, labelVar = TRUE)

# Multi-Dimension scaling plot of proximity Matrix
MDSplot(model_rf1, CD1$High)

# CONCLUSION :
# The Company data was analysed initially as a Multi Linear regression model taking Sales data as continuous data
# Then further the Sales data was converted into class variable and classification technique was applied
# Random forest algorithm was applied and accuracy was found to be 0.90
