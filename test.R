simulation <- function(sample.size = 500, p = 500, alpha=0){
  print(system.time(res <- foreach(index = 1:500,.packages = c('concordLearn'),.combine = rbind,.errorhandling='remove')%dopar%{
  ## test
  set.seed(index)
  nobs <- sample.size
  p <- p
  x <- array(rnorm(nobs*p), c(nobs, p))
  beta_true <- c(1,-1,1,-1, rep(0, times=p-4))
  beta_modify <- c(1,1,1,1, rep(0, times=p-4))
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

  # testing data
  x.test <- array(rnorm(10^4*p), c(10^4, p))
  y.test.nonoise <- 5*pnorm(x.test%*%beta_true)
  y.test <- y.test.nonoise + 0.2*rnorm(10^4)
  y.test.label <- (y.test>quantile(U)[2])
  #score.class.hinge.trans <- mean((x.test %*% fit_hinge_trans$coef > fit_hinge_trans$off.set) == y.test.label)
  #score.class.hinge <- mean((x.test %*% fit_hinge$coef > fit_hinge$off.set) == y.test.label)
  score.class.logistic.trans <- mean((x.test %*% fit_logistic_trans$coef >= fit_logistic_trans$off.set) == y.test.label)
  score.class.logistic.comb <- mean((x.test %*% fit_logistic_comb$coef >= fit_logistic_comb$off.set[1]) == y.test.label)
  score.class.logistic <- mean((x.test %*% fit_logistic$coef >= fit_logistic$off.set) == y.test.label)
  #score.rank.hinge.trans <- cor(x.test %*% fit_hinge_trans$coef,y=y.test.nonoise, method='kendall')
  #score.rank.hinge <- cor(x.test %*% fit_hinge$coef,y=y.test.nonoise, method='kendall')
  score.rank.logistic.trans <- cor(x.test %*% fit_logistic_trans$coef,y=y.test.nonoise, method='kendall')
  score.rank.logistic.comb <- cor(x.test %*% fit_logistic_comb$coef,y=y.test.nonoise, method='kendall')
  score.rank.logistic <- cor(x.test %*% fit_logistic$coef,y=y.test.nonoise, method='kendall')


  c(score=c(score.class.logistic.trans,score.class.logistic.comb,score.class.logistic,
            score.rank.logistic.trans,score.rank.logistic.comb,score.rank.logistic), pvalue=c(fit_logistic_trans$pvalue, fit_logistic_comb$pvalue, fit_logistic$pvalue))
  }))
  apply(res[,1:6], 2, mean, na.rm = TRUE)
  apply(res[,7:30],2,function(t){mean(t<0.05)})
  save(res, file=paste0("~/Simulations/concordLearn/sim","_",sample.size,"_",p,"_",alpha,"_transfer.RData"))
}

library(doParallel)
n_cores <- detectCores(all.tests = FALSE, logical = TRUE)
cl <- makeCluster(n_cores)
registerDoParallel(cl)

alpha_seq <- c(1)
for (alpha in alpha_seq){
  simulation(sample.size = 200, p=1000, alpha = alpha)
  simulation(sample.size = 350, p=1000, alpha = alpha)
  simulation(sample.size = 500, p=1000, alpha = alpha)
}


stopCluster(cl)
