correlationsAndTest <- function(matrixData, y, adjustGiven = 'bonferroni'){
  
  yNumeric <- ifelse(y == 't', 1, 0)
  
  apply(matrixData, 2, function(vector){
    if (class(vector) == 'integer' | class(vector) == 'numeric'){
      print(corr.test(as.data.frame(vector), y = as.data.frame(yNumeric),
                      adjust = adjustGiven), short = FALSE)
    }else if(class(vector) == 'factor'){
      glmModel <- glm(yNumeric ~ ., family = 'binomial', data = as.data.frame(cbind(yNumeric, vector)))
      print(glmModel)
      print(anova(glmModel))
    }
  })  
  
}