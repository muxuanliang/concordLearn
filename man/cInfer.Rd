% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/cInfer.R
\name{cInfer}
\alias{cInfer}
\title{This function implements the estimation and inference for high dimensional classification rule under a joint model which simply pool all the labels.}
\usage{
cInfer(
  x,
  y = list(y1, y2, y3),
  y_refit = NULL,
  fit = NULL,
  weight = rep(1, times = NCOL(x)),
  lossType = "logistic",
  parallel = TRUE,
  indexToTest = NULL,
  ...
)
}
\arguments{
\item{x}{a n by p matrix representing predictors, where n is the sample size and p is the number of the predictors.}

\item{y}{a list of the labels; each elment is a array of the single label.}

\item{fit}{a list contains the model fitting result from the joint modeling; fit$coef contains the coefficients (p dimension) and fit$cutoff contains J cutoff for J labels.}

\item{weight}{array of weights assigned to each observation.}

\item{lossType}{user can choose from 'logistic', 'exponential', and 'smoothed_hinge'.}

\item{parallel}{whether use parallel computing to tune parameters.}

\item{indexToTest}{the index of the coefficients the user would like to test; the default in Null.}
}
\value{

}
\description{
This function implements the estimation and inference for high dimensional classification rule under a joint model which simply pool all the labels.
}
\examples{
# generate data
nobs <- 500
p <- 500
rate <- 0
alpha <- 0
V <- function(p, rate = 0.5){
  V.matrix <- array(0, c(p,p))
  for (i in 1:p){
    for (j in 1:p){
      V.matrix[i,j] <- rate ^ (abs(i-j))
    }
  }
  V.matrix
}
x <- mgcv::rmvn(nobs, rep(0, times=p), V(p, rate))
beta_true <- c(1,-1,1,-1, rep(0, times=p-4))
beta_modify <- c(1,1,1,1, rep(0, times=p-4))
mix <- rbinom(nobs, 1, 1-alpha)
U <- sapply(pnorm(x\%*\%beta_true), function(t){
  v <- rbinom(1, 4, t)
  if (v==3){
    v.add <- rbinom(1, 1, alpha)
    v <- v+v.add
  } else if (v==4){
    v.add <- rbinom(1, 1, alpha)
    v <- v-v.add
  }
  v
})
y.cutoff <- list(U>0, U>3)
# fit use the proposed method
fit <- cInfer(x, y=y.cutoff, y_refit = list(y.cutoff[[1]]), weight = c(1, 1, 1, 1, rep(1, times= p-4)), lossType = 'logistic', tol = 1e-3, parallel = FALSE)
}
\references{
anonymous.
}
\author{
anonymous <anonymous@anonymous.net>
}
