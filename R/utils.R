# smoothed hinge with input gamma

sHinge <- function(t, order = 0, gamma = 1){
  stopifnot((order < 3) & (gamma <= 1))
  hinge <- (1-t)*((1-t)>0)
  if (order == 0){
      return((hinge^2/(2*gamma)) * (abs(hinge) <= gamma) + (abs(hinge)-gamma/2) * (abs(hinge) > gamma))
  } else if (order == 1){
    return(0 * (t>1) - 1 * (t<(1-gamma)) - (1-t)/gamma * ((1-gamma) <= t) * (t <= 1))
  } else if (order == 2){
    return(((1-gamma) <= t) * (t <= 1))
  }
}

# loss function and its derivatives

loss <- function(t, type = 'logistic', order = 0, ...){
  stopifnot((order < 3) & (type %in% c('logistic', 'exponential', 'smoothed_hinge')))
  if (order == 0){
    return(switch (type,
      'logistic' = log(1+exp(-t)),
      'exponential' = e^(-t),
      'smoothed_hinge' = sHinge(t,order=order, ...)
    ))
  } else if (order == 1){
    return(switch (type,
                   'logistic' = -exp(-t)/(1+exp(-t)),
                   'exponential' = -e^(-t),
                   'smoothed_hinge' = sHinge(t,order=order, ...)
    ))
  } else if (order == 2){
    return(switch (type,
                   'logistic' = -exp(-t)/(1+exp(-t)),
                   'exponential' = e^(-t),
                   'smoothed_hinge' = sHinge(t,order=order, ...)
    ))
  }
}

# soft.Thresh returns the soft shresholing of a vector
soft.Thresh <- function(beta, thresh){
  stopifnot(length(beta)==length(thresh))
  return((beta-thresh)*(beta > thresh) + (beta+thresh)*(beta< -thresh))
}

# solver takes data=list(X, Y) as input and output beta given a single lambda, weight, gamma
# X is the covariates, Y is the binarylabel (1 or -1), in clearn we do not incorperate the intercept but X can incoorperate the intercept.
solver <- function(x, y, lambda, betaInit = c(rep(0.1, NCOL(x))), weight = rep(1, NCOL(x)), lossType='logistic', tol = 1e-4, maxIter = 10^3, ...){
  betaSeq <- cbind(betaInit, betaInit)
  iter <- 1
  dif <- 10^3
  while((iter < maxIter) & (dif > tol)){
    ### calculate v
    v <- betaSeq[,iter+1] + (iter-2)/(iter+1) * (betaSeq[,iter+1]-betaSeq[,iter])
    ### calculate grad
    lossValue <- mean(loss(y*x %*% v, type = lossType, order = 0 , ...))
    lossDeriv <- loss(y*x %*% v, type = lossType, order = 1 , ...)
    grad <- apply(x, 2, function(t){mean(t*lossDeriv*y)})
    ### set back tracking parameter
    t <- 1
    ratio <- 0.8
    ### set first try
    newbeta <- soft.Thresh(v-t * grad, thresh = t * lambda * weight)
    newLossValue <- mean(loss(y*x %*% newbeta, type = lossType, order = 0 , ...))
    ### find appropriate step size
    while(newLossValue > (lossValue+t(grad) %*% (newbeta-v)+sum((newbeta-v)^2)/(2*t))){
      t <- t * ratio
      newbeta <- soft.Thresh(v-t * grad, thresh = t * lambda * weight)
      newLossValue <- mean(loss(y*x %*% newbeta, type = lossType, order = 0 , ...))
    }
    ### update
    dif <- max(abs(newbeta-betaSeq[,iter+1]))
    betaSeq <- cbind(betaSeq, newbeta)
    iter <- iter + 1
  }
  return(list(coef=betaSeq[,iter+1], iteration=iter))
}
