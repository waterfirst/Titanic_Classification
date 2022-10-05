# Titanic_Classification

library(tidyverse)
library(dplyr)
library(caret)
library(ModelMetrics)
library(randomForest)
library(stringr)
library(xgboost)  # for fitting GBMs
library(ranger)   # for fitting random forests
library(rpart) 
library(patchwork)
library(DataExplorer)
library(vip)

rm(list=ls())

getwd()

setwd("D:/Non_Documents/AI/R/data/220810_181047_aifactory")
dir()


train <- read.csv("train.csv", stringsAsFactors=T, na.strings=c("-999", "NA", ""))

test <- read.csv("test.csv", stringsAsFactors=T, na.strings=c("-999", "NA", ""))
submission <- read.csv("sample_submission.csv", stringsAsFactors=T)


############
# survival - 생존유무, target 값. (0 = 사망, 1 = 생존)
# pclass - 티켓 클래스. (1 = 1st, 2 = 2nd, 3 = 3rd)
# sex - 성별
# Age - 나이(세)
# sibsp - 함께 탑승한 형제자매, 배우자 수 총합
# parch - 함께 탑승한 부모, 자녀 수 총합
# ticket - 티켓 넘버
# fare - 탑승 요금
# cabin - 객실 넘버
# embarked - 탑승 항구기


head(train)
head(test)
head(submission)


# na 가 있는 열 확인하기
colSums(is.na(train))  ## Age 177개 결측치
colSums(is.na(test))  ## Age 86개, Fare 1개 결측치 


## 결측치 대체 (KNN)
library(DMwR2)
train2 <- knnImputation(train ,k=10) #KNN 대체법 실시
test2 <- knnImputation(test ,k=10) #KNN 대체법 실시
colSums(is.na(train2))
colSums(is.na(test2))  ## Age 86개, Fare 1개 결측치 

str(train2)
str(test2)



# id 열 제거하기
train2 <- train2[,-c(1,4,9,11)]
test2 <- test2[,-c(1,3, 8, 10)]


# target -> factor로 바꾸기
train2$Survived <- as.factor(train2$Survived)

str(train2)
str(test2)


pairs(train2) ## 간단히 상관관계 보기

############### GGally 패키지를 이용한 EDA  ##############################

library(ggplot2)
library(GGally)

#create pairs plot
ggpairs(train_s)

ggpairs(train_s, mapping = aes(color = continent, alpha=0.3))

###################################### EDA ##################################



####DataExplorer 패키지를 이용한 EDA

introduce(train)
plot_intro(train) ## 기본
plot_missing(train) ## NA 유무
plot_bar(train)  ## 범주형 변수 시각화화


plot_histogram(train, ncol = 3) ## 연속형 변수 시각화
plot_density(train, ncol = 3)
plot_qq(train, ncol = 3) ## 연속형 변수 정규성
plot_qq(train, by = "contract_until")  ## 그룹별로 볼 수 있음

plot_correlation(train, type = "continuous") ##상관행렬(공분산)

plot_boxplot(train[c("value", "contract_until")], by = "contract_until")

plot_prcomp(train, ggtheme = theme_bw(), variance_cap = 0.50) ## 주성분 분석
## 더미변수 만들기 (범주형 -> 연속형)
#dummify(train["wday"]) %>% head()

create_report(data = train, y="value") ## 빠른 종합 레포트



############### WVplots 패키지를 이용한 EDA  ##############################


library(WVPlots)
train_s <- train  %>% sample_n(100) %>% as_tibble()

PairPlot(train_s, colnames(train_s)[1:10], title = "aaa",
         group_var = NULL, palette="Dark2",point_color = "darkgray") 




##################################################################################

#aov 아노바 분석 (독립변수)

#install.packages("multcomp")
library(multcomp)
library(ggplot2)
cholesterol
summary(cholesterol)
aggregate(response ~ trt, data = cholesterol, mean)
ggplot(cholesterol, aes(x = trt, y = response)) +  geom_boxplot(notch = T)
result <- aov(response ~ trt, data = cholesterol)
anova(result)
plot(result, which=1:3)
shapiro.test(result$residuals)
bartlett.test(response ~ trt, data = cholesterol)
TukeyHSD(result)
par(las=2, mar=c(5, 8, 4, 2))
plot(TukeyHSD(result))

clipr::write_clip(train)   


str(train)
## Simple interaction plot
interaction.plot(x.factor     = train$continent,
                 trace.factor = train$prefer_foot,
                 response     = train$value,
                 fun = mean,
                 type="b",
                 col=c("black","red","green"),  ### Colors for levels of trace var.
                 pch=c(19, 17, 15),             ### Symbols for levels of trace var.
                 fixed=TRUE,                    ### Order by factor order in data
                 leg.bty = "o")



#PCA 하기

train_dt <- prcomp(train, center = T,scale. = T) #데이터 표준화 포함

#PCA 결과 확인
train_dt


#PCA 결과 시각화
# Proportion of variance 출력 

plot(pca_dt,type = "l")
screeplot(pca_dt, main = "", col = "green", type = "lines", pch = 1, npcs = length(pca_dt$sdev))

summary(pca_dt)







###################################### 회귀분석  ###############################



#훈련/검증 데이터  70:30 으로 나누기

idx<-createDataPartition(train2$Survived  , p=0.7, list=F)
train_df<-train2[idx,]
test_df<-train2[-idx,]


#모델 만들기
m1<-train(Survived~., data=train_df, method="glm") #로지스틱 회귀 모델

m2<-randomForest(Survived~., data=train_df, ntree=100) #랜덤포레스트 모델



# Fit a single regression tree
tree <- rpart(Survived ~ ., data = train_df)

# Fit a random forest
set.seed(101)
rfo <- ranger(Survived ~ ., data = train_df, importance = "impurity")

# Fit a GBM
set.seed(102)
bst <- xgboost(
  data = data.matrix(subset(train_df, select = -Survived)),
  label = train_df$Survived, 
  objective = "reg:linear",
  nrounds = 100, 
  max_depth = 5, 
  eta = 0.3,
  verbose = 0  # suppress printing
)



# VI plot for single regression tree
vi_tree <- tree$variable.importance
barplot(vi_tree, horiz = TRUE, las = 1)

# VI plot for RF
vi_rfo <- rfo$variable.importance %>% sort()
barplot(vi_rfo, horiz = TRUE, las = 1)

# VI plot for GMB
library(Ckmeans.1d.dp)
vi_bst <- xgb.importance(model = bst)
xgb.ggplot.importance(vi_bst)

i1 <- vip(m1) + ggtitle("Logistic regression")
i2 <- vip(m2)+ ggtitle("Random Forest")
i3 <- vip(tree)+ ggtitle("Descision tree")
i4 <- vip(rfo)+ggtitle("Fast Random Forest")
i5 <- vip(bst)+ggtitle("XGBoost")

i3+i1+i5+i2+i4


#예측하기

p1<-predict(m1, test_df)
p2<-predict(m2, test_df)
p3<-predict(tree, test_df)
p3 <- as.factor(round(p3[,2]))
p4<- predict(rfo, data = test_df, predict.all = TRUE)
p4 <- p4$predictions[,2]
p4 <- as.factor(ifelse(p4==1, 0, 1))
p5<-predict(bst, data.matrix(test_df[,-1]))
p5 <- as.factor(ifelse(round(p5)==1, 0, 1))


#평가하기 (Accuracy)

r1 <- caret::confusionMatrix(test_df$Survived, p1)$overall[1] #로지스틱 회귀분석
r2 <- caret::confusionMatrix(test_df$Survived, p2)$overall[1] #랜덤포레스트
r3 <- caret::confusionMatrix(test_df$Survived, p3)$overall[1] #의사결정나무
r4 <- caret::confusionMatrix(test_df$Survived, p4)$overall[1] #ranger
r5 <- caret::confusionMatrix(test_df$Survived, p5)$overall[1] #xgboost

name <- c("Logistic regression", "Random Forest", "Descision tree", "Fast Random Forest","XGBoost")
r_accuracy <- round(c(r1,r2,r3,r4,r5),2)
v <- as.data.frame(cbind(name, r_accuracy) )

v %>% 
  mutate(name = fct_reorder(name,desc(r_accuracy))) %>% 
  ggplot(aes(name, r_accuracy, fill=name))+geom_col() + 
  geom_text(data = v, aes(label = paste("Accuracy=",r_accuracy)), y = r_accuracy, size=5)+
  ggtitle("Titanic 생존자 예측")+
  labs(x="ML Model", y="Acuuracy", subtitle="")+
  theme_bw()+
  theme(axis.text.y = element_text(size=12), 
        axis.text.x = element_text(size=12))+
  theme(legend.position="none")

#평가하기 (F1)

f1 <- caret::confusionMatrix(test_df$Survived, p1)$byClass[7] #로지스틱 회귀분석
f2 <- caret::confusionMatrix(test_df$Survived, p2)$byClass[7]  #랜덤포레스트
f3 <- caret::confusionMatrix(test_df$Survived, p3)$byClass[7]  #의사결정나무
f4 <- caret::confusionMatrix(test_df$Survived, p4)$byClass[7]  #ranger
f5 <- caret::confusionMatrix(test_df$Survived, p5)$byClass[7]  #xgboost

name_f1 <- c("Logistic regression", "Random Forest", "Descision tree", "Fast Random Forest","XGBoost")
r_f1 <- round(c(f1,f2,f3,f4,f5),2)
v_f1 <- as.data.frame(cbind(name, r_f1) )

v_f1 %>% 
  mutate(name_f1 = fct_reorder(name,desc(r_f1))) %>% 
  ggplot(aes(name_f1, r_f1, fill=name_f1))+geom_col() + 
  geom_text(data = v_f1, aes(label = paste("F1=",r_f1)), y = r_f1-0.2, size=5)+
  ggtitle("Titanic 생존자 예측")+
  labs(x="ML Model", y="F1", subtitle="")+
  theme_bw()+
  theme(axis.text.y = element_text(size=12), 
        axis.text.x = element_text(size=12))+
  theme(legend.position="none")


#평가하기 (ROC, AUC)
library(pROC)

roc1 <-  pROC::roc(as.numeric(test_df$Survived), as.numeric(p1)) #로지스틱 회귀분석
roc2 <-  pROC::roc(as.numeric(test_df$Survived), as.numeric(p2))  #랜덤포레스트
roc3 <-  pROC::roc(as.numeric(test_df$Survived), as.numeric(p3))  #의사결정나무
roc4 <-  pROC::roc(as.numeric(test_df$Survived), as.numeric(p4))  #ranger
roc5 <-  pROC::roc(as.numeric(test_df$Survived), as.numeric(p5))  #xgboost

name_roc <- c("Logistic regression", "Random Forest", "Descision tree", "Fast Random Forest","XGBoost")
r_roc <- round(c(roc1$auc,roc2$auc, roc3$auc, roc4$auc, roc5$auc),2)
v_roc <- as.data.frame(cbind(name, r_f1) )

v_roc %>% 
  mutate(name_roc = fct_reorder(name_roc,desc(r_roc))) %>% 
  ggplot(aes(name_roc, r_roc, fill=name_roc))+geom_col() + 
  geom_text(data = v_roc, aes(label = paste("AUC=",r_roc)), y = r_roc-0.1, size=5)+
  ggtitle("Titanic 생존자 예측")+
  labs(x="ML Model", y="AUC", subtitle="")+
  theme_bw()+
  theme(axis.text.y = element_text(size=12), 
        axis.text.x = element_text(size=12))+
  theme(legend.position="none")

dev.off()
par(mfrow=c(1,5))
roc_p1 <- plot.roc(roc1,  legacy.axes = T, main="Logistic regression")
roc_p2 <- plot.roc(roc2,  legacy.axes = T, main="Random Forest")
roc_p3 <- plot.roc(roc3,  legacy.axes = T, main="Descision tree")
roc_p4 <- plot.roc(roc4,  legacy.axes = T, main="Fast Random Forest")
roc_p5 <- plot.roc(roc5,  legacy.axes = T, main="XGBoost")




## 최종 랜덤포레스트 모델로 최종 모델링 하기


m<-randomForest(Survived~., data=train2, ntree=100)

p<-predict(m, test2)

############ Importance 시각화  ################

varImpPlot(m, main="varImpPlot of iris")

library(vip)

vi(m) 
vip(m)



#데이터 제출

## p값을 문자열로 바꾸고 csv 파일로 제출하기

p<-as.character(p)
submission$Survived <- p

write.csv(submission, "submission.csv", row.names=F)

getwd()
## 제출된 값 다시 한번 확인하기

abc<-read.csv("submission.csv")

head(abc)
