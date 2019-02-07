setwd("./")
library(e1071)
library(rpart)
library(ROCR)
library(pROC)

library(corrplot)

evaluatePerformance = function(accuracy, precision, recall, f1measure){
  print(paste("Accuracy mean:",  mean(accuracy)))
  print(paste("Accuracy sd:",  sd(accuracy)))
  print(paste("Precision mean:",  mean(precision)))
  print(paste("Precision sd:",  sd(precision)))
  print(paste("Recall mean:", mean(recall)))
  print(paste("Recall sd:", sd(recall)))
  print(paste("F1measure mean:", mean(f1measure)))
  print(paste("F1measure sd:", sd(f1measure)))
}

# importazione del dataset traformato per il machine learning
dataset <- read.csv(file="Dataset/transformedML.csv", dec = ".", header=TRUE, sep=",", row.names=NULL, strip.white=TRUE)

# ANALISI ESPLORATIVA DEI DATI

dim(dataset)
sapply(dataset, class)
levels(dataset$state)

pdf("Data_exploration_plots/barlpot_service.pdf", width = 5, height = 5)
barplot(table(dataset$Service), main = "Distribuzione di diffusione del settore terziario", log = "y", col = "#FF6666")
dev.off()
pdf("Data_exploration_plots/pie_categories.pdf", width = 5, height = 5)
pie(sort(table(dataset$category), decreasing = T)[1: 10], main = "Distribuzione delle categorie delle campagne", col = c("#CCFFFF", "#FF99CC", "#FFB266", "#B2FF66", "#FF6666"))
dev.off()
pdf("Data_exploration_plots/barlpot_country.pdf", width = 5, height = 5)
barplot(table(dataset$Country), log = "y", las=2, col = "#FF6666", main = "Distribuzione di numero di campagne per nazione")
dev.off()
pdf("Data_exploration_plots/barlpot_gdp.pdf", width = 5, height = 5)
barplot(table(dataset$GDP....per.capita.), log = "y", las=2, col = "#FF6666", main = "Distribuzione del GDP")
dev.off()
table.main_category <- table(dataset$main_category)
pdf("Data_exploration_plots/pie_main_category.pdf", width = 5, height = 5)
pie(table.main_category, main = "Distribuzione delle main category", col = c("#CCFFFF", "#FF99CC", "#FFB266", "#B2FF66", "#FF6666"))
dev.off()
table.state <- table(dataset$state)
pdf("Data_exploration_plots/pie_state.pdf", width = 5, height = 5)
pie(table.state, main = "Distribuzione degli stati delle campagne", col = c("#CCFFFF", "#FF99CC", "#FFB266", "#B2FF66", "#FF6666"))
dev.off()
prop.table(table.state)

mean(dataset$backers)
sd(dataset$backers)
range(dataset$backers)
mean(dataset$goal)
sd(dataset$goal)
range(dataset$goal)

# correlazione
# serve passare alla versione numerica e normalizzata
corrdata <- dataset
corrdata$state = as.numeric(corrdata$state)
corrdata$Country = as.numeric(corrdata$Country)
corrdata$category = as.numeric(corrdata$category) 
corrdata$main_category = as.numeric(corrdata$main_category) 

for (i in 1:8){
  corrdata[,i] = (corrdata[,i] - min(corrdata[,i])) / (max(corrdata[,i]) - min (corrdata[,i]))
}

cormat <- cor(corrdata)
pdf("Data_exploration_plots/corrplot.pdf", width = 8, height = 8)
corrplot(cormat, method = "number", col="black")
dev.off()

# APPRENDIMENTO AUTOMATICO
# Randomizziamo l'ordine delle righe nel dataset
dataset <- dataset[sample(nrow(dataset)), ] 
ind <- cut(1:nrow(dataset), breaks = 10, labels = F)

# Misure: la nostra precision è quante volte prediciamo correttamente il fallimento su quante volte lo abbiamo predetto
#   recall è quante volte abbiamo predetto correttamente il fallimento su quante volte andava predetto

# BASELINE MODEL
# creiamo le lise per contenere le misure di performance
baseline.accuracy = c()
baseline.precision = c()
baseline.recall = c()
baseline.f1measure = c()
for(i in 1:10){
  trainset = dataset[ind != i, ]
  testset = dataset[ind == i, ]
  # valutiamo se nel trainset sono più diffuse istanze la cui classe 
  # è failed oppure successful e creiamo il relativo modello baseline
  tmp = prop.table(table(trainset$state))
  if(tmp["failed"] >= tmp["successful"]) {
    testset$prediction = rep("failed", nrow(testset)) 
  } else { # R sintassi non spostare l'esle!
    testset$prediction = rep("successful", nrow(testset)) 
  }
  rm(tmp)
  # verifichiamo la qualità della soluzione proposta, in modo da utilizzarla come riferimento
  # per i successivi modelli
  confusion.matrix = table(testset$state, testset$prediction)
  accuracy = sum(diag(confusion.matrix))/sum(confusion.matrix)
  baseline.accuracy = append(sum(diag(confusion.matrix))/sum(confusion.matrix), baseline.accuracy)
  precision = confusion.matrix[1,1] / (confusion.matrix[1,1] + confusion.matrix[2,1])
  baseline.precision = append(precision, baseline.precision)
  recall = confusion.matrix[1,1] / (confusion.matrix[1,1] + 0)
  baseline.recall = append(recall, baseline.recall)
  f1measure = 2 * (precision * recall / (precision + recall))
  baseline.f1measure = append(f1measure, baseline.f1measure)
  #ROC
  testset$prediction[1] = "successful"
  testset$prediction = factor(testset$prediction)
  baseline.curve = roc(testset$prediction, as.numeric(testset$state))
  baseline.auc = auc(baseline.curve)
  pdf(paste0("./AUC/baseline/auc_", i), width = 5, height = 5)
  plot.roc(baseline.curve, legacy.axes = T, col = "red", lwd = 3, asp = 1.0, main = paste0("AUC = ", round(baseline.auc, digits = 5)))
  dev.off()
}
rm(i, trainset, testset, recall, precision, f1measure, accuracy)
# Valutiamo medie e varianza delle misure di performance
print("Evaluating Baseline's performance:")
evaluatePerformance(baseline.accuracy, baseline.precision, baseline.recall, baseline.f1measure)

# DECISION TREES LIBRARIES
library(rattle)
library(rpart.plot)
library(RColorBrewer)

# DECISION TREE - NO BACKERS  
# Non utilizzando l'attributo backer, il modello mostra una capacità predittiva limitata,
# non molto distante dal modello baseline. Risulta quindi necessario eseguire delle operazioni di
# stima di questo paramentro nel caso esso non fosse disponibile.
# liste misure di performance alberi senza il parametro Backers
treeNB.accuracy = c()
treeNB.precision = c()
treeNB.recall = c()
treeNB.f1measure = c()
for(i in 1:10){
  trainset = dataset[ind != i, ]
  testset = dataset[ind == i, ]
  decisionTree = rpart(state ~ Country + GDP....per.capita. + Service + category + goal + main_category, data=trainset, method="class")
  # fancyRpartPlot(decisionTree)
  # usiamo il modello per eseguire le previsioni
  testset$prediction <- predict(decisionTree, testset, type = "class")
  # e valutiamone le performance
  confusion.matrix = table(testset$state, testset$prediction)
  accuracy = sum(diag(confusion.matrix))/sum(confusion.matrix)
  treeNB.accuracy = append(accuracy, treeNB.accuracy)
  precision = confusion.matrix[1,1] / (confusion.matrix[1,1] + confusion.matrix[2,1])
  treeNB.precision = append(precision, treeNB.precision)
  recall = confusion.matrix[1,1] / (confusion.matrix[1,1] + confusion.matrix[1,2])
  treeNB.recall = append(recall, treeNB.recall)
  f1measure = 2 * (precision * recall / (precision + recall))
  treeNB.f1measure = append(f1measure, treeNB.f1measure)
  #ROC
  treeNB.curve = roc(testset$prediction, as.numeric(testset$state))
  treeNB.auc = auc(treeNB.curve)
  pdf(paste0("./AUC/treeNB/auc_", i), width = 5, height = 5)
  plot.roc(treeNB.curve, legacy.axes = T, col = "red", lwd = 3, asp = 1.0, main = paste0("AUC = ", round(treeNB.auc, digits = 5)))
  dev.off()
}
rm(i, trainset, testset, recall, precision, f1measure, accuracy)
print("Evaluating Decision Tree without backers' performance:")
evaluatePerformance(treeNB.accuracy, treeNB.precision, treeNB.recall, treeNB.f1measure)

# DECISION TREE
# creiamo ora un albero sfruttando tutti gli attributi disponibili.
# liste misure di performance alberi non potati
tree.accuracy = c()
tree.precision = c()
tree.recall = c()
tree.f1measure = c()
# liste misure di performance alberi potati
Ptree.accuracy = c()
Ptree.precision = c()
Ptree.recall = c()
Ptree.f1measure = c()
for(i in 1:10){
  trainset = dataset[ind != i, ]
  testset = dataset[ind == i, ]
  decisionTree = rpart(state ~ ., data=trainset, method="class")
  # fancyRpartPlot(decisionTree)
  # usiamo il modello per eseguire le previsioni
  testset$prediction <- predict(decisionTree, testset, type = "class")
  # e valutiamone le performance
  confusion.matrix = table(testset$state, testset$prediction)
  accuracy = sum(diag(confusion.matrix))/sum(confusion.matrix)
  tree.accuracy = append(accuracy, tree.accuracy)
  precision = confusion.matrix[1,1] / (confusion.matrix[1,1] + confusion.matrix[2,1])
  tree.precision = append(precision, tree.precision)
  recall = confusion.matrix[1,1] / (confusion.matrix[1,1] + confusion.matrix[1,2])
  tree.recall = append(recall, tree.recall)
  f1measure = 2 * (precision * recall / (precision + recall))
  tree.f1measure = append(f1measure, tree.f1measure)
  #ROC
  tree.curve = roc(testset$prediction, as.numeric(testset$state))
  tree.auc = auc(tree.curve)
  pdf(paste0("./AUC/tree/auc_", i), width = 5, height = 5)
  plot.roc(tree.curve, legacy.axes = T, col = "red", lwd = 3, asp = 1.0, main = paste0("AUC = ", round(tree.auc, digits = 5)))
  dev.off()
  # valutiamo ora il grado di complessità dell'albero, per valutare se sia
  # opportuno eseguire una potatura dell'albero
  # printcp(decisionTree)
  # plotcp(decisionTree)

  # creiamo un albero potato e verifichiamo le nuove performance
  prunedtree = prune(decisionTree, cp=.015)
  # fancyRpartPlot(prunedtree)
  testset$prediction <- predict(prunedtree, testset, type = "class")
  confusion.matrix = table(testset$state, testset$prediction)
  accuracy = sum(diag(confusion.matrix))/sum(confusion.matrix)
  Ptree.accuracy = append(accuracy, Ptree.accuracy)
  precision = confusion.matrix[1,1] / (confusion.matrix[1,1] + confusion.matrix[2,1])
  Ptree.precision = append(precision, Ptree.precision)
  recall = confusion.matrix[1,1] / (confusion.matrix[1,1] + confusion.matrix[1,2])
  Ptree.recall = append(recall, Ptree.recall)
  f1measure = 2 * (precision * recall / (precision + recall))
  Ptree.f1measure = append(f1measure, Ptree.f1measure)
  #ROC
  Ptree.curve = roc(testset$prediction, as.numeric(testset$state))
  Ptree.auc = auc(Ptree.curve)
  pdf(paste0("./AUC/Ptree/auc_", i), width = 5, height = 5)
  plot.roc(Ptree.curve, legacy.axes = T, col = "red", lwd = 3, asp = 1.0, main = paste0("AUC = ", round(Ptree.auc, digits = 5)))
  dev.off()
}
rm(i, trainset, testset, recall, precision, f1measure, accuracy)
print("Evaluating not pruned Decision Tree's performance:")
evaluatePerformance(tree.accuracy, tree.precision, tree.recall, tree.f1measure)
print("Evaluating pruned Decision Tree's performance:")
evaluatePerformance(Ptree.accuracy, Ptree.precision, Ptree.recall, Ptree.f1measure)

# NAIVE BAYES LIBRARIES
library(e1071)

# NAIVE BAYES
# liste misure di performance
bayes.accuracy = c()
bayes.precision = c()
bayes.recall = c()
bayes.f1measure = c()
for(i in 1:10){
  trainset = dataset[ind != i, ]
  testset = dataset[ind == i, ]
  classifier <- naiveBayes(trainset, trainset$state) 
  testset$prediction = predict(classifier, testset)
  #valutiamo le performance
  confusion.matrix = table(testset$state, testset$prediction)
  accuracy = sum(diag(confusion.matrix))/sum(confusion.matrix)
  bayes.accuracy = append(accuracy, bayes.accuracy)
  precision = confusion.matrix[1,1] / (confusion.matrix[1,1] + confusion.matrix[2,1])
  bayes.precision = append(precision, bayes.precision)
  recall = confusion.matrix[1,1] / (confusion.matrix[1,1] + confusion.matrix[1,2])
  bayes.recall = append(recall, bayes.recall)
  f1measure = 2 * (precision * recall / (precision + recall))
  bayes.f1measure = append(f1measure, bayes.f1measure)
  #ROC
  bayes.curve = roc(testset$prediction, as.numeric(testset$state))
  bayes.auc = auc(bayes.curve)
  pdf(paste0("./AUC/bayes/auc_", i), width = 5, height = 5)
  plot.roc(bayes.curve, legacy.axes = T, col = "red", lwd = 3, asp = 1.0, main = paste0("AUC = ", round(bayes.auc, digits = 5)))
  dev.off()
}
rm(i, trainset, testset, recall, precision, f1measure, accuracy)
print("Evaluating Naive Bayes' performance:")
evaluatePerformance(bayes.accuracy, bayes.precision, bayes.recall, bayes.f1measure)

# saving the summary roc curve
pdf("./AUC/Mixed2", width = 5, height = 5)
plot.roc(Ptree.curve, legacy.axes = T, col = "red", lwd = 3, asp = 1.0)
plot.roc(tree.curve, legacy.axes = T, col = "orange", lwd = 3, asp = 1.0, add = T)
plot.roc(baseline.curve, legacy.axes = T, col = "purple", lwd = 3, asp = 1.0, add = T)
plot.roc(treeNB.curve, legacy.axes = T, col = "blue", lwd = 3, asp = 1.0, add = T)
plot.roc(bayes.curve, legacy.axes = T, col = "green", lwd = 3, asp = 1.0, add = T)
plot.roc(svm.curve, legacy.axes = T, col = "yellow", lwd = 3, asp = 1.0, add = T)
plot.roc(svm1000.curve, legacy.axes = T, col = "brown", lwd = 3, asp = 1.0, add = T)
legend(0.25, 0.35, legend=c("Baseline", "Tree NB", "Tree", "Pruned tree", "Naive Bayes", "SVM1", "SVM1000"), col=c("purple", "blue", "orange", "red", "green"), lty = 1, cex=0.8)
dev.off()


# I modelli successivi potrebbero richiedere ore per venir trainati. E' stato quindi deciso
# di non eseguire l'operazione di 10-fold cross validation, considerando anche la grande dimensione
# del dataset utilizzato


#SVM
library(e1071)
# creiamo un modello svm e utilizziamolo per predirre
# svm.model = svm(state ~ ., data = trainset[sample(nrow(trainset)), ], kernel = 'linear', cost = 1)
# importiamo il modello già trainato per questioni di tempo
svm.model = readRDS(file = "TOREMOVE/svm.rds")
testset = readRDS(file = "TOREMOVE/testsetsvm.rds")
print(svm.model)
svm.pred = predict(svm.model, testset) 

print("Evaluating Support Vector Machine performance with c=1:")
confusion.matrix = table(testset$state, svm.pred)
accuracy = sum(diag(confusion.matrix))/sum(confusion.matrix)
precision = confusion.matrix[1,1] / (confusion.matrix[1,1] + confusion.matrix[2,1])
recall = confusion.matrix[1,1] / (confusion.matrix[1,1] + confusion.matrix[1,2])
f1measure = 2 * (precision * recall / (precision + recall))
accuracy
precision
recall
f1measure
#ROC
svm.curve = roc(svm.pred, as.numeric(testset$state))
svm.auc = auc(svm.curve)
pdf(paste0("./AUC/svm/auc"), width = 5, height = 5)
plot.roc(svm.curve, legacy.axes = T, col = "red", lwd = 3, asp = 1.0, main = paste0("AUC = ", round(svm.auc, digits = 5)))
dev.off()

# Visto il grande numero di vettori di supporto, aumentiamo il valore di c a 100
# svm.model1000 = svm(state ~ ., data = trainset[sample(nrow(trainset)), ], kernel = 'linear', cost = 100)
# importiamo il modello già trainato
svm.model1000 = readRDS(file = "TOREMOVE/svm1000.rds")
testset1000 = readRDS(file = "TOREMOVE/testsetsvm1000.rds")
print(svm.model1000)
svm.pred1000 = predict(svm.model1000, testset1000) 

print("Evaluating Support Vector Machine performance with c=100:")

confusion.matrix = table(testset1000$state, svm.pred1000)
accuracy = sum(diag(confusion.matrix))/sum(confusion.matrix)
precision = confusion.matrix[1,1] / (confusion.matrix[1,1] + confusion.matrix[2,1])
recall = confusion.matrix[1,1] / (confusion.matrix[1,1] + confusion.matrix[1,2])
f1measure = 2 * (precision * recall / (precision + recall))
accuracy
precision
recall
f1measure
#ROC
svm1000.curve = roc(svm.pred1000, as.numeric(testset1000$state))
svm1000.auc = auc(svm.curve)
pdf(paste0("./AUC/svm1000/auc"), width = 5, height = 5)
plot.roc(svm1000.curve, legacy.axes = T, col = "red", lwd = 3, asp = 1.0, main = paste0("AUC = ", round(svm1000.auc, digits = 5)))
dev.off()


# NEURAL NETWORK
library(neuralnet)
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
# network = neuralnet(successful + failed ~  Country + GDP....per.capita. + Service + backers + category +  goal + main_category , trainsetnet, hidden=3, threshold = 0.01, stepmax = 1e+06, learningrate = 0.01)
network = readRDS(file = "TOREMOVE/network")
testsetnet = readRDS(file = "TOREMOVE/testsetnetwork")

# sfruttiamo il modello per eseguire predizione
net.predict = compute(network, testsetnet[c(1:7)])$net.result
net.prediction = c("successful", "failed")[apply(net.predict, 1, which.max)]
testsetnet$prediction = net.prediction

#valutiamo le performance
print("Evaluating Neural Network performance:")
confusion.matrix = table(testset$state, testset$prediction)
accuracy = sum(diag(confusion.matrix))/sum(confusion.matrix)
precision = confusion.matrix[1,1] / (confusion.matrix[1,1] + confusion.matrix[2,1])
recall = confusion.matrix[1,1] / (confusion.matrix[1,1] + confusion.matrix[1,2])
f1measure = 2 * (precision * recall / (precision + recall))
accuracy
precision
recall
f1measure
#ROC
nn.curve = roc(nn.pred, as.numeric(testset$state))
nn.auc = auc(nn.curve)
pdf(paste0("./AUC/nn/auc"), width = 5, height = 5)
plot.roc(nn.curve, legacy.axes = T, col = "red", lwd = 3, asp = 1.0, main = paste0("AUC = ", round(nn.auc, digits = 5)))
dev.off()
