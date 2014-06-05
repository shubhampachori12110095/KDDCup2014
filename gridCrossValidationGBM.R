gridCrossValidationGBM <- function(xGen, yGen, subset, numberOfTrees,
                                   xValFolds, coresAmount, treeDepthVector, shrinkageVector,
                                   OOBPercentage = 0.8, 
                                   plot = TRUE, distributionSelected = 'bernoulli'){
  
  #Transform factor Outcomes to 1-0 outcomes
  yGen <- ifelse(yGen == 't', 1, 0)
    
  #Cross validates features selected by the user
  grid <- expand.grid(treeDepthVector, shrinkageVector, stringsAsFactors = TRUE) #this creates all possible combinations of the elements in treeDepthVector and shrinkageVector
  
  trainError <- rep(NA, nrow(grid))
  oobError <- rep(NA, nrow(grid))
  #cvError <- rep(NA, nrow(grid))
  aucError <- rep(NA, nrow(grid))
  
  for(i in 1:nrow(grid)){
    model <- gbm.fit(x = xGen[subset, ], y = yGen[subset], 
                     n.trees = numberOfTrees, interaction.depth = grid[i, 1],
                     shrinkage = grid[i, 2], 
                     verbose = TRUE, distribution = distributionSelected,
                     nTrain = floor(length(subset) * OOBPercentage))
    
    summary(model)
    
    
    trainError[i] <- min(model$train.error)
    oobError[i] <- min(model$valid.error)
    #cvError[i] <- min(model$cv.error)
    
    #auc Error
    predictionGBM <- predict(model, newdata = xGen[-subset, ], 
                             n.trees = abs(model$n.trees - which.min(model$valid.error)), 
                             single.tree = TRUE)
    #predictionVector <- round(predictionVector)
    errorVector <- ifelse(predictionGBM > 0, 1, 0)
    aucError[i] <- auc(errorVector, yGen[-subset])  
    print(paste('Error for tree depth', grid[i, 1], 'and shrinkage', grid[i, 1], 'calculated.',
                'Out of', grid[nrow(grid), 1], 'and', grid[nrow(grid), 2], 'AUC of', aucError[i]))
  }
  
  if(plot == TRUE){
    #Plotting Errors Train Error vs. Cross Validation
    matplot(1:nrow(grid), cbind(trainError, oobError, aucError), pch = 19, col = c('red', 'blue', 'green'), type = 'b', ylab = 'Mean Squared Error(Blue, Red) + AUC(Green)', xlab = 'Tree Depth + shrinkage')
    legend('topright', legend = c('Train', 'OOB','AUC'), pch = 19, col = c('red', 'blue', 'green'))      
  } 
  
  optimalIndex <- which.min(aucError)
  
  #Return the best values found on the grid
  return(grid[optimalIndex, ])
}
