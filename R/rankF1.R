# rank auxiliary outcome according to F1-score
rankF1 <- function(x, y=list(y1, y2, y3)){
  # set sample size
  n <- NROW(x)

  # training index
  index <- sample(c(0,1), floor(n/2), replace=TRUE)

  # set y if y is not 1 or -1
  y <- lapply(y, function(t){
    t.ones <- rep(1, length(t))
    if (((levels(factor(t))[1]) != "-1")|(levels(factor(t))[2] != "1")){
      t.ones[factor(t)==levels(factor(t))[1]] <- -1
      t.ones[factor(t)==levels(factor(t))[2]] <- 1
    }
    t.ones
  })

  # fit f1 score
  f.score <- sapply(y, function(t){
    fit <- glmnet::cv.glmnet(x[index==0,], t[index==0], family='binomial')
    pred <- predict(fit, newx=x[index==1,], s=fit$lambda.min)
    MLmetrics::F1_Score(as.numeric(t[index==1]>0), as.numeric(pred>0))
  })

  return(list(f.score=f.score, rank=order(f.score, decreasing = TRUE)))
}
