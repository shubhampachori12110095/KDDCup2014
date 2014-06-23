gridCrossValidationGBM <- function(xGen, yGen, subset, numberOfTrees,
                                   xValFolds, coresAmount, treeDepthVector, shrinkageVector,
                                   OOBPercentage = 0.8, 
                                   plot = TRUE, distributionSelected = 'bernoulli',
                                   singleTreeFraction = 0.7){
  
  #libraries
  require('Metrics')
  require('gbm')
    
  #Transform factor Outcomes to 1-0 outcomes
  yGen <- ifelse(yGen == 't', 1, 0)
    
  #Cross validates features selected by the user
  grid <- expand.grid(treeDepthVector, shrinkageVector, stringsAsFactors = TRUE) #this creates all possible combinations of the elements in treeDepthVector and shrinkageVector
  
  trainError <- rep(NA, nrow(grid))
  oobError <- rep(NA, nrow(grid))
  #cvError <- rep(NA, nrow(grid))
  aucError <- rep(NA, nrow(grid))
  bestTreeVector <- rep(NA, nrow(grid))
    
  for(i in 1:nrow(grid)){
    model <- gbm.fit(x = xGen[subset, ], y = yGen[subset], 
                     n.trees = numberOfTrees, interaction.depth = grid[i, 1],
                     shrinkage = grid[i, 2], 
                     verbose = TRUE, distribution = distributionSelected,
                     nTrain = floor(length(subset) * OOBPercentage), bag.fraction = singleTreeFraction)
    
    print(gbm.perf(model, oobag.curve = TRUE, method = 'OOB'))
    
    trainError[i] <- min(model$train.error)
    oobError[i] <- min(model$valid.error)
    #cvError[i] <- min(model$cv.error)
    
    #auc Error
    predictionGBM <- predict(model, newdata = xGen[-subset, ], 
                             n.trees = which.min(model$valid.error), type = 'response')
    
    #predictionGBM <- ifelse(distributionSelected == 'bernoulli', 
    #                        break, predictionGBM <- exp(predictionGBM))
    
    aucError[i] <- auc(yGen[-subset], predictionGBM)     
    print(paste('Error for tree depth', grid[i, 1], 'with shrinkage', grid[i, 2], 'calculated.',
                  'Out of', grid[nrow(grid), 1], 'trees.', 'AUC of', aucError[i]))    
    
    #best tree
    bestTreeVector[i] <- which.min(model$valid.error)
    
  }
  
  if(plot == TRUE){
    #Plotting Errors Train Error vs. Validation
    par(mfrow=c(2, 1))
    matplot(1:nrow(grid), cbind(trainError, oobError), pch = 19, col = c('red', 'blue'), type = 'b', ylab = 'Mean Squared Error', xlab = 'Tree Depth + shrinkage')
    legend('topright', legend = c('Train', 'OOB'), pch = 19, col = c('red', 'blue'))      
    
    matplot(1:nrow(grid), aucError, pch = 19, col = 'green', type = 'b', ylab = 'AUC', xlab = 'Tree Depth + shrinkage')
    legend('topright', legend = 'AUC', pch = 19, col = 'green')    
  } 
  
  optimalIndex <- which.min(oobError)
  bestTree <- bestTreeVector[optimalIndex]
  
  #Return the best values found on the grid
  return(c(grid[optimalIndex, ], bestTree))
}
