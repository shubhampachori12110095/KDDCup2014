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
require('gbm')
require('ggplot2')
require('Metrics')

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
projectZipIndecesPartial <- charmatch(projects$school_zip[is.na(projectZipIndeces)], zip2cbsa$ZIP5)
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

#Preprocessing
projects <- transform(projects, school_city = as.factor(school_city), school_state = as.factor(school_state),
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
                      date_posted = as.Date(date_posted, format = '%Y-%m-%d')
                      )

resourcesOnProjectsTrain <- transform(resourcesOnProjectsTrain, vendorid = as.factor(vendorid), 
                                      project_resource_type = as.factor(project_resource_type) 
                                      )
resourcesOnProjectsTest <- transform(resourcesOnProjectsTest, vendorid = as.factor(vendorid), 
                                     project_resource_type = as.factor(project_resource_type)
)
                       
projectsCBSAandCSA <- as.data.frame(projectsCBSAandCSA) #they are transformed to factors automatically because the default behaivor for characters is strings as factors
names(projectsCBSAandCSA) <- c('CBSA', 'CSA')
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
#Simple Validation Projects Model
#GBM
nTrainingSamples <- 50000
trainIndicesy <- sample(1:length(y), nTrainingSamples) # Number of samples considered for prototyping / best parameter selection, it has to be greater than 500 the sampling size, otherwise it will throw an error saying that more data is required 
#trainIndicesy <- sample(1:length(y), length(y)) # Use this line to use the complete dataset and shuffle the data

indicesProjectsShuffled <- indicesTrainProjects[trainIndicesy]
indicesEssaysShuffled <- indicesTrainEssays[trainIndicesy]
indicesResourcesShuffled <- indicesTrainResources[trainIndicesy]

  
#Setting cross validation parameters
amountOfTrees <- 20000
NumberofCVFolds <- 5
cores <- NumberofCVFolds

if (NumberofCVFolds > 3){
  cores <- detectCores() - 1
}

treeDepth <- 7 #interaction.depth validation

#sample indices
set.seed(102)
sampleIndices <- sort(sample(1:length(indicesProjectsShuffled), floor(length(indicesProjectsShuffled) * 0.8))) # these indices are useful for validation

#project variables' indices
variablesIndices <- c(8, 10, seq(13, 34)) # with all of the valid variables
  
##grid cross validation using OOB Error
OptimalValidationGBMValues <- gridCrossValidationGBM(xGen = projects[indicesProjectsShuffled, variablesIndices], 
                                                     yGen = y[trainIndicesy], sampleIndices, amountOfTrees,
                                                     NumberofCVFolds, cores,
                                                     seq(3, 5), 0.003)

OptimalValidationGBMValues <- gridCrossValidationGBM(xGen = cbind(projects[indicesProjectsShuffled, variablesIndices], 
                                                                  essaysLength[indicesEssaysShuffled]), 
                                                     yGen = y[trainIndicesy], sampleIndices, amountOfTrees,
                                                     NumberofCVFolds, cores,
                                                     seq(3, 5), 0.003)

OptimalValidationGBMValues <- gridCrossValidationGBM(xGen = cbind(projects[indicesProjectsShuffled, variablesIndices],
                                                                  essaysLength[indicesEssaysShuffled], 
                                                                  projectsCBSAandCSA[indicesProjectsShuffled, ]), 
                                                     yGen = y[trainIndicesy], sampleIndices, amountOfTrees,
                                                     NumberofCVFolds, cores,
                                                     c(1, 3, 5), c(0.001, 0.003))

OptimalValidationGBMValues <- gridCrossValidationGBM(xGen = cbind(projects[indicesProjectsShuffled, variablesIndices],
                                                                  essaysLength[indicesEssaysShuffled], 
                                                                  projectsCBSAandCSA[indicesProjectsShuffled, ], 
                                                                  resourcesOnProjectsTrain[indicesResourcesShuffled, ]), 
                                                     yGen = y[trainIndicesy], sampleIndices, amountOfTrees,
                                                     NumberofCVFolds, cores,
                                                     c(1, 3, 5), c(0.001, 0.003))

#optimal hipeparameters for tree depth and for shrinkage
optimalTreeDepth <- OptimalValidationGBMValues[1]
optimalShrinkage <- OptimalValidationGBMValues[2]
bestTree <- OptimalValidationGBMValues[3]

#Use best hiperparameters
trainIndicesy <- sample(1:length(y), length(y)) # Use this line to use the complete dataset and shuffle the data

#project variables' indices
variablesIndices <- c(8, 10, seq(13, 34)) # with all of the valid variables

indicesProjectsShuffled <- indicesTrainProjects[trainIndicesy]
indicesEssaysShuffled <- indicesTrainEssays[trainIndicesy]
indicesResourcesShuffled <- indicesTrainResources[trainIndicesy]

yGen <- ifelse(y == 't', 1, 0)
GBMModel <- gbm.fit(x = cbind(projects[indicesProjectsShuffled, variablesIndices],
                              essaysLength[indicesEssaysShuffled], 
                              projectsCBSAandCSA[indicesProjectsShuffled, ]),
                    y = yGen[trainIndicesy], n.trees = 5000, interaction.depth = optimalTreeDepth,
                    shrinkage = optimalShrinkage, verbose = TRUE, distribution = 'bernoulli'
                    )

summary(GBMModel)

# check performance using OOB Error
#best.iter <- gbm.perf(GBMModel, method= 'train', plot.it = TRUE, oobag.curve = TRUE)
#print(best.iter)

#Prediction
predictionGBM <- predict(GBMModel, newdata = cbind(projects[indicesTestProjects, variablesIndices], 
                                                   essaysLength[indicesTestEssays], 
                                                   projectsCBSAandCSA[indicesTestProjects, ]), 
                         n.trees = bestTree, single.tree = TRUE, type = 'response')


predictionGBMOverloaded <- predict(GBMModel, newdata = cbind(projects[indicesTestProjects, variablesIndices], 
                                                             essaysLength[indicesTestEssays], 
                                                             projectsCBSAandCSA[indicesTestProjects, ]),
                                   n.trees = bestTree + 2000, single.tree = TRUE, type = 'response')

#Save .csv file 
submissionTemplate$is_exciting <- predictionGBM
write.csv(submissionTemplate, file = "predictionI.csv", row.names = FALSE, quote = FALSE)
