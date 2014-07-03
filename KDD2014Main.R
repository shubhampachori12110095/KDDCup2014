#KDD Cup 2014
#No more initial comments now that I know there are people reading this
#version 0.1

#########################
#Init
rm(list=ls(all=TRUE))

#Load/install libraries
require('plyr')
require('tm')
require('psych')
require('Matrix')
require('glmnet')
require('leaps')
require('gbm')
require('ggplot2')
require('Metrics')
require('forecast')

#Set Working Directory
workingDirectory <- '~/Wacax/Kaggle/KDD Cup 2014/KDD Cup 2014/'
setwd(workingDirectory)

dataDirectory <- '~/Wacax/Kaggle/KDD Cup 2014/Data/'

#Load functions
source(paste0(workingDirectory, 'text2Matrix.R'))
source(paste0(workingDirectory, 'gridCrossValidationGBM.R'))
source(paste0(workingDirectory, 'extractBestTree.R'))
source(paste0(workingDirectory, 'correlationsAndTest.R'))

#############################
#Load Data
#Input Data

#Essay Length
essayNames <- names(read.csv(paste0(dataDirectory, 'essays.csv'), nrows = 1000, stringsAsFactors = FALSE))
essayClasses <- sapply(read.csv(paste0(dataDirectory, 'essays.csv'), nrows = 1000, stringsAsFactors = FALSE), class)

essays <- read.csv(paste0(dataDirectory, 'essays.csv'), header = TRUE, 
                   colClasses = essayClasses, col.names = essayNames,
                   stringsAsFactors = FALSE)

essaysLength <- with(essays, nchar(essay)) 
save(essaysLength, file = 'essaysLength.RData')
rm(essays)

#Essay Bag-Of Words
numberOfDivisions <- 1000 #change this to make bigger or smaller fragments/chunks to analyze later
numberOfEssays <- 664098 #do not change unless there is a new dataset
rowsToRead <- numberOfEssays / numberOfDivisions
if(numberOfEssays %% numberOfDivisions == 0){
  rowsToRead <- rep(rowsToRead, numberOfDivisions)
}else{
  rowsToRead <- c(rep(ceiling(rowsToRead), numberOfEssays %% numberOfDivisions), rep(floor(rowsToRead), numberOfDivisions - numberOfEssays %% numberOfDivisions))
}
#function testing
ifelse(sum(rowsToRead) == numberOfEssays, print(paste('Processing', rowsToRead[1], 'essays at a time')), 
       print(paste('function will not compute all essays. Leaving out', numberOfEssays - sum(rowsToRead), 'essays')))

#create here the loop
#essaysCorpora <- text2Matrix('dummy corpora', sparse = TRUE,  Sparsity = 0.9)
#for(i in 1:numberOfDivisions){
for(i in 1:2){
  essays <- read.csv(paste0(dataDirectory, 'essays.csv'), header = ifelse(i == 1, TRUE, FALSE), 
                   nrows = rowsToRead[i], skip = ifelse(i == 1, 0, sum(rowsToRead[1:i - 1])), 
                   colClasses = essayClasses, col.names = essayNames,
                   stringsAsFactors = FALSE)
  
  #essaysCorpora <- c(essaysCorpora, text2Matrix(essays$need_statement, sparse = TRUE, Sparsity = 0.9))
  #if(i == 1){essaysCorpora <- essaysCorpora[-1, ]}
  print(paste(i, 'of', numberOfDivisions))
}

#essaysCorpora <- weightTfIdf(essaysCorpora)

#Train and Test
projects <- read.csv(paste0(dataDirectory, 'projects.csv'), header = TRUE, stringsAsFactors = FALSE, na.strings=c("", "NA", "NULL"))
resources <- read.csv(paste0(dataDirectory, 'resources.csv'), header = TRUE, stringsAsFactors = FALSE, na.strings=c("", "NA", "NULL"))

#Just Train
outcomes <- read.csv(paste0(dataDirectory, 'outcomes.csv'), header = TRUE, stringsAsFactors = FALSE, na.strings=c("", "NA", "NULL"))
donations <- read.csv(paste0(dataDirectory, 'donations.csv'), header = TRUE, stringsAsFactors = FALSE, na.strings=c("", "NA", "NULL"))

#Template
submissionTemplate <- read.csv(paste0(dataDirectory, 'sampleSubmission.csv'), header = TRUE, stringsAsFactors = FALSE, na.strings=c("", "NA", "NULL"))

#Extra Data
zip2cbsaNames <- names(read.table(paste0(dataDirectory, 'zip07_cbsa06.txt'), header = TRUE,
                                  stringsAsFactors = FALSE, sep = ',', colClasses = "character",
                                  nrows = 100, na.strings=c("", "NA", "NULL")))

zip2cbsa <- read.table(paste0(dataDirectory, 'zip07_cbsa06.txt'), header = TRUE,
                       stringsAsFactors = FALSE, sep = ',', colClasses = "character",
                       na.strings=c("", "NA", "NULL"), col.names = zip2cbsaNames)

################################################################
#Outcomes to predict
y <- as.factor(outcomes$is_exciting)
save(y, file = 'y.RData')

#Create Train and Test dataframes
#Projects Indices Train 
indicesTrainProjects <- match(outcomes$projectid, projects$projectid)
save(indicesTrainProjects, file = 'indicesTrainProjects.RData')
#resources Indices Train 
indicesTrainResources <- match(outcomes$projectid, resources$projectid)
save(indicesTrainResources, file = 'indicesTrainResources.RData')
#Essays Indices Train
indicesTrainEssays <- match(outcomes$projectid, essays$projectid)
save(indicesTrainEssays, file = 'indicesTrainEssays.RData')

#Indices Test
#Projects
indicesTestProjects <- match(submissionTemplate$projectid, projects$projectid)
save(indicesTestProjects, file = 'indicesTestProjects.RData')
#Resources
indicesTestResources <- match(submissionTemplate$projectid, resources$projectid)
save(indicesTestResources, file = 'indicesTestResources.RData')
#Essays
indicesTestEssays <- match(submissionTemplate$projectid, essays$projectid)
save(indicesTestEssays, file = 'indicesTestEssays.RData')

#Additional Feature Engineering
projectZipIndeces <- match(projects$school_zip, zip2cbsa$ZIP5)
#WARNING
#These two lines remove NAs in the indices but replace the indices with partial matches
#--------------------------------------------------------------------------------------
projectZipIndecesPartial <- pmatch(projects$school_zip[is.na(projectZipIndeces)], zip2cbsa$ZIP5)
projectZipIndeces[is.na(projectZipIndeces)] <- projectZipIndecesPartial
#--------------------------------------------------------------------------------------
projectsCBSAandCSA <- cbind(zip2cbsa$CBSA.CODE[projectZipIndeces], zip2cbsa$CSA.CODE[projectZipIndeces])
save(projectsCBSAandCSA, file = 'projectsCBSAandCSA.RData')

#resources' features onto the projects data
#WARNING
#These four lines remove NAs in the indices but replace the indices with partial matches
#UPDATE, they won't find a partial match
#--------------------------------------------------------------------------------------
PartialIndicesTrain <- pmatch(outcomes$projectid[is.na(indicesTrainResources)], resources$projectid)
indicesTrainResources[is.na(indicesTrainResources)] <- PartialIndicesTrain
PartialIndicesTest <- pmatch(outcomes$projectid[is.na(indicesTestResources)], resources$projectid)
indicesTestResources[is.na(indicesTestResources)] <- PartialIndicesTest
#--------------------------------------------------------------------------------------
#Resources Train
resourcesOnProjectsTrain <- resources[indicesTrainResources, c('vendorid', 'project_resource_type', 
                                                                 'item_unit_price', 'item_quantity')]
save(resourcesOnProjectsTrain, file = 'resourcesOnProjectsTrain.RData')
#Resources Test
resourcesOnProjectsTest <- resources[indicesTestResources, c('vendorid', 'project_resource_type', 
                                                               'item_unit_price', 'item_quantity')]
save(resourcesOnProjectsTest, file = 'resourcesOnProjectsTest.RData')

#------------------------
#Time Series Analysis
#Additional column containing only the month when the project was posted
projects$YearMonth <- strftime(projects$date_posted, format = '%Y-%m')
#Aggregate occurrences to create a monthly frecuence
positiveFrecuencies <- ddply(projects[indicesTrainProjects[y == 't'], ], .(YearMonth), nrow)
totalFrecuencies <- ddply(projects[indicesTrainProjects, ], .(YearMonth), nrow)
names(positiveFrecuencies) <- c('YearMonth', 'ExcitingProjects')
names(totalFrecuencies) <- c('YearMonth', 'TotalProjects')
par(mfrow=c(2, 1))
ggplot(data = positiveFrecuencies, aes(x = YearMonth, y = ExcitingProjects, group = 1)) +  geom_line() +  geom_point()
ggplot(data = totalFrecuencies, aes(x = YearMonth, y = TotalProjects, group = 1)) +  geom_line() +  geom_point()

# from 2010 to Dec 2013 as a time series object
myts <- ts(positiveFrecuencies[,2], start=c(2010, 4), end=c(2013, 12), frequency = 12)
#from 2002 to Dec 2013 all training dataa as a time series object
mytsAll <- ts(totalFrecuencies[,2], start=c(2002, 9), end=c(2013, 12), frequency = 12)

# plot series, ggplot above is nicer
plot(myts)
plot(mytsAll)
# Seasonal decompostion
plot(stl(myts, s.window="period"))
plot(stl(mytsAll, s.window="period"))
# additional plots
monthplot(myts)
monthplot(mytsAll)
seasonplot(myts) 
seasonplot(mytsAll) 

# Automated forecasting using an exponential model
#fit <- ets(myts)
# Automated forecasting using an ARIMA model
fit <- auto.arima(myts)
fitFull <- auto.arima(mytsAll)

modelForecast <- forecast(fit, 5)
modelForecastFull <- forecast(fitFull, 5)
plot(modelForecast)
plot(modelForecastFull)

#Performing a Backcast to generate simulated probabilities before May 2010 where there are no exciting projects
# Reverse time
revmyts <- ts(rev(myts), frequency=12)
# Backcast
bc <- forecast(auto.arima(revmyts), abs(length(myts) - length(mytsAll)))
plot(bc)
# Reverse time again
bc$mean <- ts(rev(bc$mean), end=tsp(myts)[1] - 1/12, frequency=12)
bc$upper <- bc$upper[abs(length(myts) - length(mytsAll)):1,]
bc$lower <- bc$lower[abs(length(myts) - length(mytsAll)):1,]
bc$x <- myts
# Plot result
plot(bc, xlim=c(tsp(myts)[1]-abs(length(myts) - length(mytsAll))/12, tsp(myts)[2]))

positiveFrecuencies <- c(as.numeric(bc$mean), positiveFrecuencies[,2], as.numeric(modelForecast$mean))
positiveFrecuencies[positiveFrecuencies<0] <- 0
totalFrecuencies <- c(totalFrecuencies[,2], as.numeric(modelForecastFull$mean))

positiveProbs <- as.data.frame(cbind(sort(unique(projects$YearMonth)), 
                                    positiveFrecuencies/totalFrecuencies, 
                                    positiveFrecuencies, 
                                    totalFrecuencies))

names(positiveProbs) <- c('YearMonth', 'positiveProb', 'PositiveFreq', 'TotalFreq')
positiveProbs$positiveProb[positiveProbs$positiveProb<=0] <- 0.00001
ggplot(data = positiveProbs, aes(x = YearMonth, y = positiveProb, group = 1)) +  geom_line() +  geom_point()

#CreAte extra columns in projects data; containing: probabilities of being positive, positive frequencies and total frecuencies
projects <- merge(projects, positiveProbs, by = 'YearMonth')
save(projects, file = 'projects.RData')
#-------------------------
#Preprocessing
projects <- transform(projects, YearMonth = as.factor(YearMonth), school_city = as.factor(school_city), school_state = as.factor(school_state),
                      school_metro = as.factor(school_metro), school_charter = as.factor(school_charter), 
                      school_magnet = as.factor(school_magnet), school_year_round = as.factor(school_year_round), 
                      school_nlns = as.factor(school_nlns), school_kipp = as.factor(school_kipp), 
                      school_charter_ready_promise = as.factor(school_charter_ready_promise), 
                      teacher_prefix = as.factor(teacher_prefix), teacher_teach_for_america = as.factor(teacher_teach_for_america),
                      teacher_ny_teaching_fellow = as.factor(teacher_ny_teaching_fellow), primary_focus_subject = as.factor(primary_focus_subject),
                      primary_focus_area = as.factor(primary_focus_area), secondary_focus_subject = as.factor(secondary_focus_subject),
                      secondary_focus_area = as.factor(secondary_focus_area), resource_type = as.factor(resource_type),
                      poverty_level = as.factor(poverty_level), grade_level = as.factor(grade_level),
                      fulfillment_labor_materials = as.factor(fulfillment_labor_materials),
                      eligible_double_your_impact_match = as.factor(eligible_double_your_impact_match), eligible_almost_home_match = as.factor(eligible_almost_home_match),
                      date_posted = as.Date(date_posted, format = '%Y-%m-%d'), 
                      positiveProb = as.numeric(positiveProb), PositiveFreq = as.numeric(PositiveFreq), 
                      TotalFreq = as.numeric(TotalFreq)
                      )


resourcesOnProjectsTrain <- transform(resourcesOnProjectsTrain, vendorid = as.factor(vendorid), 
                                      project_resource_type = as.factor(project_resource_type) 
                                      )
resourcesOnProjectsTest <- transform(resourcesOnProjectsTest, vendorid = as.factor(vendorid), 
                                     project_resource_type = as.factor(project_resource_type)
)
                       
projectsCBSAandCSA <- as.data.frame(projectsCBSAandCSA, stringsAsFactors = FALSE) #they are transformed to factors automatically because the default behaivor for characters is strings as factors
names(projectsCBSAandCSA) <- c('CBSA', 'CSA')
projectsCBSAandCSA[is.na(projectsCBSAandCSA[ ,'CBSA']), 'CBSA'] <- 'rural'
projectsCBSAandCSA[is.na(projectsCBSAandCSA[ ,'CSA']), 'CSA'] <- 'rural'
projectsCBSAandCSA <- transform(projectsCBSAandCSA, CBSA = as.factor(CBSA), 
                                      CSA = as.factor(CSA))

#################################################################
#EDA
#Unique Samples
ggplot(as.data.frame(y), aes(y)) + geom_histogram()
isExitingProbabilities <- table(y) / length(y)

str(projects)
apply(projects, 2, function(vector){return(length(unique(vector)))})
str(resources)
apply(resources, 2, function(vector){return(length(unique(vector)))})

#Find correlations between data
rowProjects <- 50000
#rowProjects <- nrow(projects)
correlationsProjectsList <- correlationsAndTest(projects[indicesTrainProjects[1:rowProjects], c(30, 31, 32)], y[1:rowProjects])
correlationsResourcesList <- correlationsAndTest(resources[indicesTrainProjects[1:rowProjects], c(30, 31, 32)], y[1:rowProjects])

#####################################################################
#Predictors Selection
#NA omit, regsubsets is sensitive to NAs
noNAIndices <- which(apply(is.na(projects[indicesTrainProjects, ]), 1, sum) == 0)
#projectsNoNAs  <- na.omit(projects[indicesTrainProjects, ])

#project variables' indices in projects
variablesIndicesFull <- c(9, 11, seq(14, 35), 37, 38, 39) # with all of the valid variables
#resourcesIndicesFull <- c(2, 3)

#sample data
nTrainingSamples <- 240210 #this is just the max number of samples the auc function can handle
trainIndicesy <- sample(1:length(y[noNAIndices]), nTrainingSamples) # Number of samples considered for prototyping / best parameter selection, it has to be greater than 500 the sampling size, otherwise it will throw an error saying that more data is required 
#trainIndicesy <- sample(1:y[noNAIndices], length(y[noNAIndices])) # Use this line to use the complete dataset and shuffle the data

#indicesProjectsShuffled <- indicesTrainProjects[indicesValidTrain][trainIndicesy]
#indicesEssaysShuffled <- indicesTrainEssays[trainIndicesy]
#indicesResourcesShuffled <- indicesTrainResources[trainIndicesy]


with(na.omit(projects[indicesTrainProjects, ])[ , variablesIndicesFull], 
     sum(is.na(school_metro)))

#y transform to 1-0 probabilities
yGen <- ifelse(y == 't', 1, 0)

#Forward Stepwise Selection
projectsFit <- regsubsets(yGen ~ . , 
                          data = cbind(na.omit(projects[indicesTrainProjects, ])[trainIndicesy , variablesIndicesFull], 
                                       yGen[noNAIndices[trainIndicesy]]),
                          method = "forward", na.action="na.exclude")

#projectsFit2 <- glmulti(yGen ~ . , 
#                       data = cbind(na.omit(projects[indicesTrainProjects, ])[trainIndicesy , variablesIndicesFull], 
#                                    yGen[noNAIndices[trainIndicesy]]),
#                       method = "g")

#projectsFit3 <- glmnet(x = as.matrix(na.omit(projects[indicesTrainProjects, ])[trainIndicesy , variablesIndicesFull]),
#                       y = yGen[noNAIndices[trainIndicesy]], family = 'binomial')

projectsFit4 <- gbm.fit(x = cbind(na.omit(projects[indicesTrainProjects, ])[trainIndicesy , variablesIndicesFull], essaysLength[indicesTrainEssays[noNAIndices]][trainIndicesy], projectsCBSAandCSA[indicesTrainEssays[noNAIndices], ][trainIndicesy, ]),
                       y = yGen[noNAIndices[trainIndicesy]],  n.trees = 2500, interaction.depth = 7,
                       shrinkage = 0.001, verbose = TRUE, distribution = 'bernoulli', 
                       bag.fraction = 0.65, nTrain = floor(nTrainingSamples * 0.7))

projectsFit5 <- gbm.fit(x = cbind(na.omit(projects[indicesTrainProjects, ])[trainIndicesy , variablesIndicesFull], essaysLength[indicesTrainEssays[noNAIndices]][trainIndicesy]),
                        y = yGen[noNAIndices[trainIndicesy]],  n.trees = 2500, interaction.depth = 7,
                        shrinkage = 0.001, verbose = TRUE, distribution = 'bernoulli', 
                        bag.fraction = 0.65, nTrain = floor(nTrainingSamples * 0.7))

predict4 <- predict.gbm(projectsFit4, newdata = cbind(na.omit(projects[indicesTrainProjects, ])[-trainIndicesy , variablesIndicesFull], essaysLength[indicesTrainEssays[noNAIndices]][-trainIndicesy], projectsCBSAandCSA[indicesTrainEssays[noNAIndices], ][-trainIndicesy, ]),
                   n.trees = which.min(projectsFit4$valid.error), type = 'response')
predict5 <- predict.gbm(projectsFit5, newdata = cbind(na.omit(projects[indicesTrainProjects, ])[-trainIndicesy , variablesIndicesFull], essaysLength[indicesTrainEssays[noNAIndices]][-trainIndicesy]),
                        n.trees = which.min(projectsFit5$valid.error), type = 'response')
print(paste('AUC error with CBSAs of:', auc(yGen[noNAIndices[-trainIndicesy]][1:100000], predict4[1:100000])))
print(paste('AUC error without CBSAs of:', auc(yGen[noNAIndices[-trainIndicesy]][1:100000], predict5[1:100000])))
print(paste('OOB Error with CBSAs', min(projectsFit4$valid.error)))
print(paste('OOB Error without CBSAs', min(projectsFit5$valid.error)))

sum4 <- summary(projectsFit4)
sum5 <- summary(projectsFit5)

bestTree4 <- gbm.perf(projectsFit4, method = 'OOB')
gbm.perf(projectsFit4, method = 'OOB', oobag.curve = TRUE)
bestTree5 <- gbm.perf(projectsFit5, method = 'OOB')
gbm.perf(projectsFit5, method = 'OOB', oobag.curve = TRUE)

plot(projectsFit4, type = 'response', n.trees = bestTree4)
plot(projectsFit5, type = 'response', n.trees = bestTree5)

#best predictors with CBSA
variablesIndices4 <- as.character(sum4$var[sum4$rel.inf > 0.01])
projectPredictorsCBSA <- na.omit(match(variablesIndices4, names(projects)))
#best predictors without CBSA nor CSA
variablesIndices5 <- as.character(sum4$var[sum4$rel.inf > 0.01])
projectPredictorsNOCBSA <- na.omit(match(variablesIndices5, names(projects)))
projectPredictors <- intersect(projectPredictorsCBSA, projectPredictorsNOCBSA)

#####################################################################
#Simple Validation Projects Model
#NAs indices
noNAIndices <- which(apply(is.na(projects[indicesTrainProjects, ]), 1, sum) == 0)
#projectsNoNAs  <- na.omit(projects[indicesTrainProjects, ])

#valid predictors in projects (indices/columns)
print(projectPredictors)

#---------------------------------------
#GBM
#sample data
#Without NAs
nTrainingSamples <- 100000 #this is just the max number of samples the auc function can handle
trainIndicesy <- sample(1:length(y[noNAIndices]), nTrainingSamples) # Number of samples considered for prototyping / best parameter selection, it has to be greater than 500 the sampling size, otherwise it will throw an error saying that more data is required 
#trainIndicesy <- sample(1:length(y[noNAIndices]), length(y[noNAIndices])) # Use this line to use the complete dataset and shuffle the data
#With NAs
nTrainingSamples <- 100000
#trainIndicesy <- sample(1:length(y), nTrainingSamples) # Number of samples considered for prototyping / best parameter selection, it has to be greater than 500 the sampling size, otherwise it will throw an error saying that more data is required 
trainIndicesy <- sample(1:length(y), length(y)) # Use this line to use the complete dataset and shuffle the data

#shuffle data with NAs
indicesProjectsShuffled <- indicesTrainProjects[trainIndicesy]
indicesEssaysShuffled <- indicesTrainEssays[trainIndicesy]
indicesResourcesShuffled <- indicesTrainResources[trainIndicesy]
#sample indices with and without NAs
set.seed(102)
sampleIndices <- sort(sample(1:length(trainIndicesy), floor(length(trainIndicesy) * 0.8))) # these indices are useful for validation

#Setting cross validation parameters
amountOfTrees <- 3000
NumberofCVFolds <- 5

treeDepth <- 7 #interaction.depth validation

##grid cross validation using OOB Error with NAs
#OptimalValidationGBMValues <- gridCrossValidationGBM(xGen = projects[indicesProjectsShuffled, projectPredictors], 
#                                                     yGen = y[trainIndicesy], sampleIndices, amountOfTrees,
#                                                     NumberofCVFolds, cores,
#                                                     distributionSelected = 'adaboost',
#                                                     seq(1, treeDepth, 3), c(0.001, 0.003))

OptimalValidationGBMValues2 <- gridCrossValidationGBM(xGen = cbind(projects[indicesProjectsShuffled, projectPredictors], 
                                                                  essaysLength[indicesEssaysShuffled]), 
                                                     yGen = y[trainIndicesy], sampleIndices, amountOfTrees,
                                                     NumberofCVFolds, cores,
                                                     distributionSelected = 'bernoulli',
                                                     seq(1, treeDepth, 3), c(0.001, 0.003))

OptimalValidationGBMValues3 <- gridCrossValidationGBM(xGen = cbind(projects[indicesProjectsShuffled, projectPredictors],
                                                                  essaysLength[indicesEssaysShuffled], 
                                                                  projectsCBSAandCSA[indicesProjectsShuffled, 'CBSA']), 
                                                     yGen = y[trainIndicesy], sampleIndices, amountOfTrees,
                                                     distributionSelected = 'bernoulli',
                                                     NumberofCVFolds, cores,
                                                     seq(1, treeDepth, 3), c(0.001, 0.003))

#OptimalValidationGBMValues4 <- gridCrossValidationGBM(xGen = cbind(projects[indicesProjectsShuffled, projectPredictors],
#                                                                  essaysLength[indicesEssaysShuffled], 
#                                                                  projectsCBSAandCSA[indicesProjectsShuffled, 'CBSA'], 
#                                                                  resourcesOnProjectsTrain[indicesResourcesShuffled, ]), 
#                                                     yGen = y[trainIndicesy], sampleIndices, amountOfTrees,
#                                                     distributionSelected = 'adaboost',
#                                                     NumberofCVFolds, cores,
#                                                     seq(1, treeDepth, 3), c(0.001, 0.003))

#optimal hipeparameters for tree depth and for shrinkage with NAs, without CBSAs
optimalTreeDepthNoCBSA <- OptimalValidationGBMValues2[1]
optimalShrinkageNoCBSA <- OptimalValidationGBMValues2[2]
bestTreeNoCBSA <- OptimalValidationGBMValues2[3]

#optimal hipeparameters for tree depth and for shrinkage with NAs, with CBSAs
optimalTreeDepth <- OptimalValidationGBMValues3[1]
optimalShrinkage <- OptimalValidationGBMValues3[2]
bestTree <- OptimalValidationGBMValues3[3]


##grid cross validation using OOB Error without NAs
OptimalValidationGBMValNoNAs <- gridCrossValidationGBM(xGen = cbind(na.omit(projects[indicesTrainProjects, ])[trainIndicesy , projectPredictors], essaysLength[indicesTrainEssays[noNAIndices]][trainIndicesy], projectsCBSAandCSA[indicesTrainEssays[noNAIndices], ][trainIndicesy, 1]),                                                                   
                                                       yGen = y[noNAIndices[trainIndicesy]], sampleIndices, amountOfTrees,
                                                       NumberofCVFolds, cores,
                                                       distributionSelected = 'bernoulli',
                                                       seq(1, treeDepth, 3), c(0.001, 0.003))

#optimal hipeparameters for tree depth and for shrinkage without NAs
optimalTreeDepthNoNAS <- OptimalValidationGBMValNoNAs[1]
optimalShrinkageNoNAS <- OptimalValidationGBMValNoNAs[2]
bestTreeNoNAS <- OptimalValidationGBMValNoNAs[3]

#--------------------------------------#
#NAs indices
noNAIndices <- which(apply(is.na(projects[indicesTrainProjects, ]), 1, sum) == 0)

#Model creation
trainIndicesyNoNA <- sample(1:length(y[noNAIndices]), length(y[noNAIndices])) # Use this line to use the complete dataset and shuffle the data

#Use best hiperparameters
trainIndicesy <- sample(1:length(y), length(y)) # Use this line to use the complete dataset and shuffle the data

#project variables' indices
print(projectPredictors)
variablesIndices <- projectPredictors # with all of the important variables

indicesProjectsShuffled <- indicesTrainProjects[trainIndicesy]
indicesEssaysShuffled <- indicesTrainEssays[trainIndicesy]
indicesResourcesShuffled <- indicesTrainResources[trainIndicesy]

yGen <- ifelse(y == 't', 1, 0)
GBMModel <- gbm.fit(x = cbind(projects[indicesProjectsShuffled, variablesIndices],
                              essaysLength[indicesEssaysShuffled], 
                              projectsCBSAandCSA[indicesProjectsShuffled, 'CBSA']),
                    y = yGen[trainIndicesy], n.trees = bestTree, interaction.depth = optimalTreeDepth,
                    shrinkage = optimalShrinkage, verbose = TRUE, distribution = 'bernoulli',
                    bag.fraction = 0.65
                    )

GBMModelNoCBSA <- gbm.fit(x = cbind(projects[indicesProjectsShuffled, variablesIndices],
                              essaysLength[indicesEssaysShuffled]),
                    y = yGen[trainIndicesy], n.trees = bestTreeNoCBSA, interaction.depth = optimalTreeDepthNoCBSA,
                    shrinkage = optimalShrinkageNoCBSA, verbose = TRUE, distribution = 'bernoulli',
                    bag.fraction = 0.65
)

GBMModelNoNAs <- gbm.fit(x = cbind(na.omit(projects[indicesTrainProjects, ])[trainIndicesyNoNA , projectPredictors], essaysLength[indicesTrainEssays[noNAIndices]][trainIndicesyNoNA], projectsCBSAandCSA[indicesTrainEssays[noNAIndices], ][trainIndicesyNoNA, 1]),
                         y = yGen[noNAIndices[trainIndicesyNoNA]], n.trees = bestTreeNoNAS, 
                         interaction.depth = optimalTreeDepthNoNAS,
                         shrinkage = optimalShrinkageNoNAS, verbose = TRUE, distribution = 'bernoulli',
                         bag.fraction = 0.65
)

GBMModelNoNAsNoCBSA <- gbm.fit(x = cbind(na.omit(projects[indicesTrainProjects, ])[trainIndicesyNoNA , projectPredictors], essaysLength[indicesTrainEssays[noNAIndices]][trainIndicesyNoNA]),
                               y = yGen[noNAIndices[trainIndicesyNoNA]], n.trees = bestTreeNoNAS, 
                               interaction.depth = optimalTreeDepthNoNAS,
                               shrinkage = optimalShrinkageNoNAS, verbose = TRUE, distribution = 'bernoulli',
                               bag.fraction = 0.65
)

summary.gbm(GBMModel)
plot.gbm(GBMModel)
pretty.gbm.tree(GBMModel, i.tree = bestTree)

#Prediction
predictionGBM <- predict(GBMModel, newdata = cbind(projects[indicesTestProjects, variablesIndices], 
                                                   essaysLength[indicesTestEssays], 
                                                   projectsCBSAandCSA[indicesTestProjects, 'CBSA']), 
                         n.trees = bestTree, type = 'response')

predictionNoCBSAGBM <- predict(GBMModelNoCBSA, newdata = cbind(projects[indicesTestProjects, variablesIndices], 
                                                   essaysLength[indicesTestEssays]), 
                         n.trees = bestTreeNoCBSA, type = 'response')


predictionGBMNoNAs <- predict(GBMModelNoNAs, newdata = cbind(projects[indicesTestProjects, variablesIndices], 
                                                             essaysLength[indicesTestEssays], 
                                                             projectsCBSAandCSA[indicesTestProjects, 'CBSA']),
                                   n.trees = bestTreeNoNAS , type = 'response')

predictionGBMNoNAsNoCBSA <- predict(GBMModelNoNAsNoCBSA, newdata = cbind(projects[indicesTestProjects, variablesIndices], 
                                                                         essaysLength[indicesTestEssays]),
                                    n.trees = bestTreeNoNAS , type = 'response')

#Save .csv file 
submissionTemplate$is_exciting <- predictionGBM
write.csv(submissionTemplate, file = "predictionXX.csv", row.names = FALSE, quote = FALSE)

#Save .csv file 
submissionTemplate$is_exciting <- predictionNoCBSAGBM
write.csv(submissionTemplate, file = "predictionXXI.csv", row.names = FALSE, quote = FALSE)

#Save .csv file 
submissionTemplate$is_exciting <- predictionGBMNoNAs
write.csv(submissionTemplate, file = "predictionXXII.csv", row.names = FALSE, quote = FALSE)

#Save .csv file 
submissionTemplate$is_exciting <- predictionGBMNoNAsNoCBSA
write.csv(submissionTemplate, file = "predictionXXIII.csv", row.names = FALSE, quote = FALSE)
