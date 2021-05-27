#' This function implements the estimation and inference for high dimensional classification rule under a joint model which simply pool all the labels.
#'
#' @param x a n by p matrix representing predictors, where n is the sample size and p is the number of the predictors.
#' @param y a list of the labels; each elment is a array of the single label.
#' @param fit a list contains the model fitting result from the joint modeling; fit$coef contains the coefficients (p dimension) and fit$cutoff contains J cutoff for J labels.
#' @param weight array of weights assigned to each observation.
#' @param lossType user can choose from 'logistic', 'exponential', and 'smoothed_hinge'.
#' @param parallel whether use parallel computing to tune parameters.
#' @param indexToTest the index of the coefficients the user would like to test; the default in Null.
#' @return A list
#' \describe{
#' \item{fit}{a list contain details on the fitting procedure, i.e. cross-validation results, coef for each parameter choice.}
#' \item{coef}{coefficients of the estimated decision rule}
#' \item{off.set}{cut off of the \code{predictor %*% coef}}
#' \item{indexToTest}{indexToTest in the input}
#' \item{pvalue}{pvalues for the coef indexed in teh indexToTest}
#' }
#'@examples
#' # generate data
#' nobs <- 500
#' p <- 500
#' rate <- 0
#' alpha <- 0
#' V <- function(p, rate = 0.5){
#'   V.matrix <- array(0, c(p,p))
#'   for (i in 1:p){
#'     for (j in 1:p){
#'       V.matrix[i,j] <- rate ^ (abs(i-j))
#'     }
#'   }
#'   V.matrix
#' }
#' x <- mgcv::rmvn(nobs, rep(0, times=p), V(p, rate))
#' beta_true <- c(1,-1,1,-1, rep(0, times=p-4))
#' beta_modify <- c(1,1,1,1, rep(0, times=p-4))
#' mix <- rbinom(nobs, 1, 1-alpha)
#' U <- sapply(pnorm(x%*%beta_true), function(t){
#'   v <- rbinom(1, 4, t)
#'   if (v==3){
#'     v.add <- rbinom(1, 1, alpha)
#'     v <- v+v.add
#'   } else if (v==4){
#'     v.add <- rbinom(1, 1, alpha)
#'     v <- v-v.add
#'   }
#'   v
#' })
#' y.cutoff <- list(U>0, U>3)
#' # fit use the proposed method
#' fit <- cInfer(x, y=y.cutoff, y_refit = list(y.cutoff[[1]]), weight = c(1, 1, 1, 1, rep(1, times= p-4)), lossType = 'logistic', tol = 1e-3, parallel = FALSE)
#' @author anonymous <anonymous@anonymous.net>
#' @references anonymous.
#' @export

cInfer <- function(x, y=list(y1, y2, y3), y_refit = NULL, fit = NULL, weight = rep(1, times=NCOL(x)), lossType='logistic', parallel = TRUE, indexToTest = NULL, ...){
  # set how any outcomes
  n.cutoff <- length(y)

  # set sample size
  n <- NROW(x)

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
  x.combine <- cbind(apply(-pracma::eye(n.cutoff), 1, function(t){pracma::repmat(t, n, 1)}), pracma::repmat(x, n.cutoff, 1))
  y.combine <- unlist(y)



  # if coef=NULL
  if (is.null(fit)){
    # if unly one outcome
    if (n.cutoff == 1){
      fit <- cv.cLearn(x=x, y=y.combine, lambdaSeq = NULL, weight = weight, lossType = lossType, parallel = parallel, intercept=TRUE, ...)
      coef <- fit$fit$coef[,fit$lambda.seq==fit$lambda.opt]
      if(lossType == "logistic") off.set <- -fit$fit$a0[fit$lambda.seq==fit$lambda.opt]
      else off.set <- -fit$fit$coef[(1:n.cutoff),fit$lambda.seq==fit$lambda.opt]

      return(refitInfer(x=x, y=y.combine, refit = list(coef=coef, off.set=off.set), lossType=lossType, parallel = parallel, indexToTest = indexToTest, ...))
    } else {
      fit <- cv.cLearn(x=x.combine, y=y.combine, lambdaSeq = NULL, weight = c(rep(0, times=length(y)), weight), lossType = lossType, parallel = parallel, ...)
      # set coef and off.set
      coef <- fit$fit$coef[-(1:n.cutoff),fit$lambda.seq==fit$lambda.opt]
      off.set <- fit$fit$coef[(1:n.cutoff),fit$lambda.seq==fit$lambda.opt]
    }
  } else {
    # set coef and off.set
    coef <- fit$coef
    off.set <- fit$cutoff
  }

  if (!is.null(y_refit)){
    # set y_refit
    y_refit <- lapply(y_refit, function(t){
      t.ones <- rep(1, length(t))
      if (((levels(factor(t))[1]) != "-1")|(levels(factor(t))[2] != "1")){
        t.ones[factor(t)==levels(factor(t))[1]] <- -1
        t.ones[factor(t)==levels(factor(t))[2]] <- 1
      }
      t.ones
    })

    # change to one cutoff
    n.cutoff <- 1
    x.refit <- cbind(x %*% coef, x)
    y.refit <- unlist(y_refit)

    # refit
    fit_refit <- cv.cLearn(x=x.refit, y=y.refit, lambdaSeq = NULL, weight = c(0, weight), lossType = lossType, parallel = parallel, intercept=TRUE, ...)
    coef <- fit_refit$fit$coef[-1,fit_refit$lambda.seq==fit_refit$lambda.opt]+fit_refit$fit$coef[1,fit_refit$lambda.seq==fit_refit$lambda.opt]*coef
    if(lossType == "logistic") off.set <- -fit_refit$fit$a0[fit_refit$lambda.seq==fit_refit$lambda.opt]
    else off.set <- -fit_refit$fit$coef[(1:n.cutoff),fit_refit$lambda.seq==fit_refit$lambda.opt]


    return(refitInfer(x=x, y=y.refit, refit = list(coef=coef, off.set=off.set), lossType=lossType, parallel = parallel, indexToTest = indexToTest, ...))
  }

  # if indextoTest is null
  if (is.null(indexToTest)){
    indexToTest <- c(1:8)
  }

  # calculate link and first order derivative
  link <- matrix(apply(pracma::repmat(x %*% coef, 1, n.cutoff), 1, function(t){t-off.set}), nrow = n, ncol = n.cutoff, byrow = TRUE)
  first.order <- array(0, c(n, n.cutoff))
  for (i in 1:n.cutoff){
    first.order[,i] <- y[[i]] * loss(y[[i]] * link[,i], lossType=lossType, order=1, ...)
  }
  second.order <- array(0, c(n, n.cutoff))
  for (i in 1:n.cutoff){
    second.order[,i] <- loss(y[[i]] * link[,i], lossType=lossType, order=2, ...)
  }

  # fit w
  fit_w <- NULL
  score <- rep(NA, times=length(indexToTest))
  sigma <- rep(NA, times=length(indexToTest))
  sigmaNULL <- rep(NA, times=length(indexToTest))
  coefAN <- rep(NA, times=length(indexToTest))
  I <- rep(NA, times=length(indexToTest))
  if (!parallel){
    for (index in 1:length(indexToTest)){
      # set pseudo data
      pseudo.x <- cbind(apply(x[, -indexToTest[index]], 2, function(t){t * apply(first.order, 1, sum)}), -first.order)
      pseudo.y <- x[,indexToTest[index]] * apply(first.order, 1, sum)
      #fit w
      fit_w[[index]] <- glmnet::cv.glmnet(x=pseudo.x, y=pseudo.y, intercept = FALSE, standardize = FALSE)
      link_w <- predict(fit_w[[index]], newx = pseudo.x, s=fit_w[[index]]$lambda.min)
      # set beta null
      coefNULL <- coef
      coefNULL[indexToTest[index]] <- 0
      # get score under null
      linkNULL <- matrix(apply(pracma::repmat(x %*% coefNULL, 1, n.cutoff), 1, function(t){t-off.set}), nrow = n, ncol = n.cutoff, byrow = TRUE)
      first.orderNULL <- array(0, c(n, n.cutoff))
      for (i in 1:n.cutoff){
        first.orderNULL[,i] <- y[[i]] * loss(y[[i]] * linkNULL[,i], lossType=lossType, order=1, ...)
      }
      pseudo.xNULL <- cbind(apply(x[, -indexToTest[index]], 2, function(t){t * apply(first.orderNULL, 1, sum)}), -first.orderNULL)
      pseudo.yNULL <- x[,indexToTest[index]] * apply(first.orderNULL, 1, sum)
      link_wNULL <- predict(fit_w[[index]], newx = pseudo.xNULL, s=fit_w[[index]]$lambda.min)
      tmpNULL <- pseudo.yNULL-link_wNULL
      score[index] <- mean(tmpNULL)
      sigmaNULL[index] <- sqrt(mean((tmpNULL)^2))

      # set betaAN
      tmp <- pseudo.y-link_w
      Itmp <- (x[,indexToTest[index]])^2 * apply(second.order, 1, sum) - predict(fit_w[[index]], newx = cbind(apply(x[, -indexToTest[index]], 2, function(t){t * apply(second.order, 1, sum) * x[,indexToTest[index]]}), -apply(second.order, 2, function(t){t * x[,indexToTest[index]]})), s=fit_w[[index]]$lambda.min)
      coefAN[index] <- coef[indexToTest[index]]-mean(tmp) /(mean(Itmp))
      sigma[index] <- sqrt(mean((tmp)^2))
      I[index] <- (mean(Itmp))
    }
  } else {
    # set multi-thread
    library(doParallel)
    n_cores <- detectCores(all.tests = FALSE, logical = TRUE)
    cl <- makeCluster(min(length(indexToTest), n_cores))
    registerDoParallel(cl)

    #calculate
    res <- foreach(index=indexToTest,.packages = 'glmnet') %dopar%{
      # set pseudo data
      pseudo.x <- cbind(apply(x[, -indexToTest[index]], 2, function(t){t * apply(first.order, 1, sum)}), -first.order)
      pseudo.y <- x[,indexToTest[index]] * apply(first.order, 1, sum)
      #fit w
      fit_w <- glmnet::cv.glmnet(x=pseudo.x, y=pseudo.y, intercept = FALSE, standardize = FALSE)
      link_w <- predict(fit_w, newx = pseudo.x, s=fit_w$lambda.min)
      # set beta null
      coefNULL <- coef
      coefNULL[indexToTest[index]] <- 0
      # get score under null
      linkNULL <- matrix(apply(pracma::repmat(x %*% coefNULL, 1, n.cutoff), 1, function(t){t-off.set}), nrow = n, ncol = n.cutoff, byrow = TRUE)
      first.orderNULL <- array(0, c(n, n.cutoff))
      for (i in 1:n.cutoff){
        first.orderNULL[,i] <- y[[i]] * loss(y[[i]] * linkNULL[,i], lossType=lossType, order=1, ...)
      }
      pseudo.xNULL <- cbind(apply(x[, -indexToTest[index]], 2, function(t){t * apply(first.orderNULL, 1, sum)}), -first.orderNULL)
      pseudo.yNULL <- x[,indexToTest[index]] * apply(first.orderNULL, 1, sum)
      link_wNULL <- predict(fit_w, newx = pseudo.xNULL, s=fit_w$lambda.min)
      tmpNULL <- pseudo.yNULL-link_wNULL
      score <- mean(tmpNULL)
      sigmaNULL <- sqrt(mean((tmpNULL)^2))

      # set betaAN
      tmp <- pseudo.y-link_w
      Itmp <- (x[,indexToTest[index]])^2 * apply(second.order, 1, sum) - predict(fit_w, newx = cbind(apply(x[, -indexToTest[index]], 2, function(t){t * apply(second.order, 1, sum) * x[,indexToTest[index]]}), -apply(second.order, 2, function(t){t * x[,indexToTest[index]]})), s=fit_w$lambda.min)
      coefAN <- coef[indexToTest[index]]-mean(tmp) /(mean(Itmp))
      sigma <- sqrt(mean((tmp)^2))
      I <- (mean(Itmp))
      list(fit_w = fit_w, score=score, sigmaNULL=sigmaNULL, sigma=sigma, coefAN=coefAN, I=I)
    }
    stopCluster(cl)
    for (index in indexToTest){
      score[index] <- res[[index]]$score
      sigmaNULL[index] <- res[[index]]$sigmaNULL
      sigma[index] <- res[[index]]$sigma
      coefAN[index] <- res[[index]]$coefAN
      I[index] <- res[[index]]$I
    }
  }
  list(fit=fit, coef=coef, off.set=off.set, indexToTest=indexToTest, pvalue=pnorm(-abs(sqrt(n)*score/sigmaNULL))*2, coefAN=coefAN, sigmaAN=sigma/I)
}
