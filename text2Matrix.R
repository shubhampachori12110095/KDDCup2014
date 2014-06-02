text2Matrix <- function(corpus, Sparsity = 0.8, dictionary = FALSE, sparse = TRUE, weightTfIdf = FALSE){
  
  #Libraries
  require(tm)
  require(SnowballC)  
  
  #Use TM Package to pre-process corpus
  corpus <- Corpus(VectorSource(corpus))
  corpus <- tm_map(corpus, function(x) iconv(enc2utf8(x), sub = "byte"))
  corpus <- tm_map(corpus, stripWhitespace)
  corpus <- tm_map(corpus, tolower)
  corpus <- tm_map(corpus, removeWords, union(stopwords("en"), stopwords("SMART")))
  corpus <- tm_map(corpus, removePunctuation)
  corpus <- tm_map(corpus, removeNumbers)
  corpus <- tm_map(corpus, stemDocument)
  
  if (dictionary == FALSE & sparse == TRUE){
    
    ifelse(weightTfIdf == TRUE, 
           return(weightTfIdf(removeSparseTerms(DocumentTermMatrix(corpus), sparse = Sparsity))), 
           return(removeSparseTerms(DocumentTermMatrix(corpus), sparse = Sparsity))
           )
    
  }else if(dictionary == FALSE & sparse == FALSE){
    
    return(as.matrix(weightTfIdf(removeSparseTerms(DocumentTermMatrix(corpus), sparse = Sparsity))))
    
  }else{
    
    return(as.matrix(DocumentTermMatrix(corpus, list(dictionary = dictionary))))
    
  }
  
}