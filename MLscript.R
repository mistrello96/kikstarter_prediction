setwd("~/Projects/kickstarterprediction/")
library(e1071)
library(rpart)

# importazione del dataset traformato per il machine learning
dataset <- read.csv(file="Dataset/transformedML.csv", dec = ".", header=TRUE, sep=",", row.names=NULL, strip.white=TRUE)

# ANALISI ESPLORATIVA DEI DATI

dim(dataset)
sapply(dataset, class)
levels(dataset$state)

x <- dataset[, c(2,3,4,6)]
par(mfrow=c(1,4))
for(i in 1:4) {
  boxplot(x[,i], main=names(x)[i]) }
rm(x)
rm(i)

mean(dataset$backers)
sd(dataset$backers)
range(dataset$backers)
mean(dataset$goal)
sd(dataset$goal)
range(dataset$goal)

table.country <- table(dataset$Country)
sort(table.country,decreasing=TRUE)[1:3]
table.category <- table(dataset$category)
sort(table.category,decreasing=TRUE)[1:3]
table.main_category <- table(dataset$main_category)
pie(table.main_category)
table.state <- table(dataset$state)
pie(table.state)
prop.table(table.state)

# creazione degli indici per la divisione tra trainset e testset
ind = sample(2, nrow(dataset), replace = TRUE, prob=c(0.7, 0.3))
trainset = dataset[ind == 1,]
testset = dataset[ind == 2,]
rm(ind)

# BASELINE MODEL
# Visto che la maggior parte dei progetti è risultata fallimentare, 
# abbiamo deciso di creare un modello ingenuo che risponda sempre fallito ad ogni sample sottomesso

testset$prediction = rep("failed", 93618)

# verifichiamo la qualità della soluzione proposta, in modo da utilizzarla come riferimento
# per i successivi modelli
confusion.matrix = table(testset$state, testset$prediction)
accuracy = sum(diag(confusion.matrix))/sum(confusion.matrix)
precision = confusion.matrix[1,1] / (confusion.matrix[1,1] + 0)
recall = confusion.matrix[1,1] / (confusion.matrix[1,1] + confusion.matrix[2,1])
f1measure = 2 * (precision * recall / (precision + recall))

# DECISION TREE
library(rattle)
library(rpart.plot)
library(RColorBrewer)
# Non utilizzando l'attributo backer, il modello mostra una capacità predittiva limitata,
# non molto distante dal modello baseline. Risulta quindi necessario eseguire delle operazioni di
# stima di questo paramentro nel caso esso non fosse disponibile
decisionTree = rpart(state ~ Country + GDP....per.capita. + Service + category + goal + main_category, data=trainset, method="class")
fancyRpartPlot(decisionTree)

# creiamo ora un albero sfruttando tutti gli attributi disponibili
decisionTree = rpart(state ~ ., data=trainset, method="class")
fancyRpartPlot(decisionTree)

# usiamo il modello per eseguire le previsioni
testset$prediction <- predict(decisionTree, testset, type = "class")

# e valutiamone le performance
confusion.matrix = table(testset$state, testset$prediction)
accuracy = sum(diag(confusion.matrix))/sum(confusion.matrix)
precision = confusion.matrix[1,1] / (confusion.matrix[1,1] + 0)
recall = confusion.matrix[1,1] / (confusion.matrix[1,1] + confusion.matrix[2,1])
f1measure = 2 * (precision * recall / (precision + recall))

# valutiamo ora il grado di complessità dell'albero, per valutare se sia
# opportuno eseguire una potatura dell'albero
printcp(decisionTree)
plotcp(decisionTree)

# creiamo un albero potato e verifichiamo le nuove performance
prunedtree = prune(decisionTree, cp=.015)
fancyRpartPlot(prunedtree)

testset$prediction <- predict(prunedtree, testset, type = "class")

confusion.matrix = table(testset$state, testset$prediction)
accuracy = sum(diag(confusion.matrix))/sum(confusion.matrix)
precision = confusion.matrix[1,1] / (confusion.matrix[1,1] + 0)
recall = confusion.matrix[1,1] / (confusion.matrix[1,1] + confusion.matrix[2,1])
f1measure = 2 * (precision * recall / (precision + recall))

# NEURAL NETWORK
library(neuralnet)
library(textir)
# è necessario modificare il train e il test set in modo che le variabili non numeriche siano
# rappresentate da un numero, in quanto l'algoritmo di learning delle reti neurali non prevede
# la presenza di feature non numeriche
trainsetnet <- trainset
trainsetnet$successful = trainset$state == "successful"
trainsetnet$failed = trainset$state == "failed"

trainsetnet$Country = as.numeric(trainset$Country)
trainsetnet$category = as.numeric(trainset$category) 
trainsetnet$main_category = as.numeric(trainset$main_category) 

for (i in 1:7){
  trainsetnet[,i] = (trainsetnet[,i] - min(trainsetnet[,i])) / (max(trainsetnet[,i]) - min (trainsetnet[,i]))
}

testsetnet <- testset
testsetnet$Country = as.numeric(testset$Country) 
testsetnet$category = as.numeric(testset$category) 
testsetnet$main_category = as.numeric(testset$main_category) 

for (i in 1:7){
  testsetnet[,i] = (testsetnet[,i] - min(testsetnet[,i])) / (max(testsetnet[,i]) - min (testsetnet[,i]))
}

# creiamo la rete e la addestriamo
network = neuralnet(successful + failed ~  Country + GDP....per.capita. + Service + backers + category +  goal + main_category , trainsetnet[c(0:1000), ], hidden=3)
#network = neuralnet(successful + failed ~  GDP....per.capita. + Service + backers + goal , trainsetnet[c(0:1000), ], hidden = 3)
#network

# sfruttiamo il modello per eseguire predizione
net.predict = compute(network, testsetnet[c(1:7)])$net.result
net.prediction = c("successful", "failed")[apply(net.predict, 1, which.max)]
testsetnet$prediction = net.prediction

#valutiamo le performance
confusion.matrix = table(testset$state, testset$prediction)
accuracy = sum(diag(confusion.matrix))/sum(confusion.matrix)
precision = confusion.matrix[1,1] / (confusion.matrix[1,1] + 0)
recall = confusion.matrix[1,1] / (confusion.matrix[1,1] + confusion.matrix[2,1])
f1measure = 2 * (precision * recall / (precision + recall))

