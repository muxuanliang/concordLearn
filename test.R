simulation <- function(sample.size = 500, p = 500, alpha=0, rate=0){
  print(system.time(res <- foreach(index = 1:500,.packages = c('concordLearn'),.combine = rbind,.errorhandling='remove')%dopar%{
  ## test
  set.seed(index)
  nobs <- sample.size

  V <- function(p, rate = 0.5){
    V.matrix <- array(0, c(p,p))
    for (i in 1:p){
      for (j in 1:p){
        V.matrix[i,j] <- rate ^ (abs(i-j))
      }
    }
    V.matrix
  }

  p <- p
  x <- mgcv::rmvn(sample.size, rep(0, times=p), V(p, rate))
  if (rate>0) {
    x[, 4*(1:floor(p/4))] <- as.numeric(x[, 4*(1:floor(p/4))]>0)
  }
  beta_true <- c(1,-1,1,-1, rep(0, times=4), 0.5,-0.5,2,-2,0.5,0.5, rep(0, times=p-14))
  beta_modify <- c(1,1,1,1, rep(0, times=4), 0.5,-0.5,2,-2,0.5,0.5, rep(0, times=p-14))
  U <- 5*pnorm(x%*%beta_true) + 0.2*rnorm(nobs)
  y1 <- U
  U1 <-  (5*pnorm(x%*%beta_modify) + 0.2*rnorm(nobs))
  y2 <- (1-alpha) * U+alpha * U1
  y.cutoff <- list(y1>quantile(U)[2], y2>quantile(y2)[3])

  # concord score
  #(mean(y.cutoff[[1]]*y.cutoff[[2]])-mean(y.cutoff[[1]])*mean(y.cutoff[[2]]))/var(y.cutoff[[2]])

  ## fit using smoothed_hinge
  #system.time(fit_hinge_trans <- cInfer(x, y=y.cutoff, y_refit = list(y>cutoff[1]), weight = c(1, 1, 1, 1, rep(1, times= p-4)), lossType = 'smoothed_hinge', tol = 1e-3, parallel = FALSE))
  #system.time(fit_hinge <- cInfer(x, y=list(y>cutoff[1]), weight = c(1, 1, 1, 1, rep(1, times= p-4)), lossType = 'smoothed_hinge', tol = 1e-3, parallel = FALSE))

  ## fit using logistic
  system.time(fit_logistic_trans <- cInfer(x, y=y.cutoff, y_refit = list(y.cutoff[[1]]), weight = c(1, 1, 1, 1, rep(1, times= p-4)), lossType = 'logistic', tol = 1e-3, parallel = FALSE))
  system.time(fit_logistic_comb <- cInfer(x, y=y.cutoff, weight = c(1, 1, 1, 1, rep(1, times= p-4)), lossType = 'logistic', tol = 1e-3, parallel = FALSE))
  system.time(fit_logistic <- cInfer(x, y=list(y.cutoff[[1]]), weight = c(1, 1, 1, 1, rep(1, times= p-4)), lossType = 'logistic', tol = 1e-3, parallel = FALSE))
  system.time(fit_logistic_trans_sai <- simTrans(x, y=y.cutoff, y_refit = list(y.cutoff[[1]]), weight = c(1, 1, 1, 1, rep(1, times= p-4)), lossType = 'logistic', tol = 1e-3, parallel = FALSE))
  fit_logistic_mtl1 <- multiTask(x, y=y.cutoff)

  # testing data
  x.test <- mgcv::rmvn(10^4, rep(0, times=p), V(p, rate))
  if (rate>0) {
    x.test[, 4*(1:floor(p/4))] <- as.numeric(x.test[, 4*(1:floor(p/4))]>0)
  }
  y.test.nonoise <- 5*pnorm(x.test%*%beta_true)
  y.test <- y.test.nonoise + 0.2*rnorm(10^4)
  y.test.label <- (y.test>quantile(U)[2])
  #score.class.hinge.trans <- mean((x.test %*% fit_hinge_trans$coef > fit_hinge_trans$off.set) == y.test.label)
  #score.class.hinge <- mean((x.test %*% fit_hinge$coef > fit_hinge$off.set) == y.test.label)
  #score.class.logistic.mtl2 <- mean((x.test %*% fit_logistic_mtl2$W[,1] >= -fit_logistic_mtl2$C[1]) == y.test.label)
  score.class.logistic.trans <- mean((x.test %*% fit_logistic_trans$coef >= fit_logistic_trans$off.set) == y.test.label)
  score.class.logistic.comb <- mean((x.test %*% fit_logistic_comb$coef >= fit_logistic_comb$off.set[1]) == y.test.label)
  score.class.logistic <- mean((x.test %*% fit_logistic$coef >= fit_logistic$off.set) == y.test.label)
  #score.rank.hinge.trans <- cor(x.test %*% fit_hinge_trans$coef,y=y.test.nonoise, method='kendall')
  #score.rank.hinge <- cor(x.test %*% fit_hinge$coef,y=y.test.nonoise, method='kendall')
  #score.rank.logistic.mtl2 <- cor(x.test %*% fit_logistic_mtl2$W[,1],y=y.test.nonoise, method='kendall')
  score.rank.logistic.trans <- cor(x.test %*% fit_logistic_trans$coef,y=y.test.nonoise, method='kendall')
  score.rank.logistic.comb <- cor(x.test %*% fit_logistic_comb$coef,y=y.test.nonoise, method='kendall')
  score.rank.logistic <- cor(x.test %*% fit_logistic$coef,y=y.test.nonoise, method='kendall')

  score.class.logistic.mtl1 <- mean((x.test %*% fit_logistic_mtl1$coef >= fit_logistic_mtl1$cutoff) == y.test.label)
  score.class.logistic.trans.sai <- mean((x.test %*% fit_logistic_trans_sai$coef >= fit_logistic_trans_sai$off.set) == y.test.label)
  score.rank.logistic.mtl1 <- cor(x.test %*% fit_logistic_mtl1$coef,y=y.test.nonoise, method='kendall')
  score.rank.logistic.trans.sai <- cor(x.test %*% fit_logistic_trans_sai$coef,y=y.test.nonoise, method='kendall')


  c(score=c(score.class.logistic.trans,score.class.logistic.mtl1,score.class.logistic.trans.sai,score.class.logistic.comb,score.class.logistic,
            score.rank.logistic.trans,score.rank.logistic.mtl1,score.rank.logistic.trans.sai,score.rank.logistic.comb,score.rank.logistic), pvalue=c(fit_logistic_trans$pvalue, fit_logistic_comb$pvalue, fit_logistic$pvalue))
  }))
  print(apply(res[,1:10], 2, mean, na.rm = TRUE))
  print(apply(res[,11:34],2,function(t){mean(t<0.05)}))
  save(res, file=paste0("/mnt/c/Users/lmx19/Documents/Simulations/concordLearn/sim","_",sample.size,"_",p,"_",alpha,"_", rate,"_transfer.RData"))
}

library(doParallel)
n_cores <- detectCores(all.tests = FALSE, logical = TRUE)
cl <- makeCluster(n_cores)
registerDoParallel(cl)

rate <- 0

alpha_seq <- c(0.75,1)
for (alpha in alpha_seq){
  simulation(sample.size = 200, p=1000, alpha = alpha, rate=rate)
  simulation(sample.size = 350, p=1000, alpha = alpha, rate=rate)
  simulation(sample.size = 500, p=1000, alpha = alpha, rate=rate)
}

rate <- 0.5

alpha_seq <- c(0,0.25,0.5,0.75,1)
for (alpha in alpha_seq){
  simulation(sample.size = 200, p=1000, alpha = alpha, rate=rate)
  simulation(sample.size = 350, p=1000, alpha = alpha, rate=rate)
  simulation(sample.size = 500, p=1000, alpha = alpha, rate=rate)
}


stopCluster(cl)
