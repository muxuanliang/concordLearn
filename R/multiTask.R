multiTask <-function(x, y=list(y1, y2, y3)){
  # set how any outcomes
  n.cutoff <- length(y)

  # set sample size
  n <- NROW(x)
  p <- NCOL(x)

  # set y if y is not 1 or -1
  y <- lapply(y, function(t){
    t.ones <- rep(1, length(t))
    if (((levels(factor(t))[1]) != "-1")|(levels(factor(t))[2] != "1")){
      t.ones[factor(t)==levels(factor(t))[1]] <- -1
      t.ones[factor(t)==levels(factor(t))[2]] <- 1
    }
    t.ones
  })

  # form a training matrix
  x.combine <- pracma::eye(n.cutoff) %x% cbind(1,x)
  y.combine <- unlist(y)

  # fit
  fit <- gglasso::cv.gglasso(x=x.combine, y=y.combine, group=rep(1:(p+1), times=n.cutoff), loss='logit', pred.loss = 'loss')

  list(coef=fit$gglasso.fit$beta[c(2:(p+1)),fit$lambda==fit$lambda.min], cutoff=-(fit$gglasso.fit$beta[1,fit$lambda==fit$lambda.min]+fit$gglasso.fit$b0[fit$lambda==fit$lambda.min]))
}
