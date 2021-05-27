# Can a joint model assist target label prediction? Conditions and approaches

Description: The package implements the proposed method used in the simulation and application to heart failure readmission prediction in this paper.

License: MIT

## Requirements

This is an R package and requires R (>= 4.0.1)

Imports: 
         glmnet

Encoding: UTF-8

## Install R package

To install the R package, run this command:

```install
R CMD INSTALL 
```

## Synthetic Dataset

To evaluate the proposed method by synethetic dataset, run the following commend in R:

```eval
source("test_count.R")
```

## Example

The following example shows hwo to use the package and the main functrion cInfer. Details can be found in the help document in R.

### Synethetic data generation
``` Synethetic data generation
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
U <- sapply(pnorm(x%*%beta_true), function(t){
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
```
### Fit using the proposed method
``` fit using proposed method
fit <- cInfer(x, y=y.cutoff, y_refit = list(y.cutoff[[1]]), weight = c(1, 1, 1, 1, rep(1, times= p-4)), lossType = 'logistic', tol = 1e-3, parallel = FALSE))
```

## Results

Our model achieves the following performance in the application to heart failure readmission prediction:

| Model name         | Accuracy  | Se |
| ------------------ |---------------- | -------------- |
| Proposed   |     76.1%         |     7.18*1e-4       |

