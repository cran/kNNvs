% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/kNNsv.R
\name{kNNvs}
\alias{kNNvs}
\title{k Nearest Neighbors with Grid Search Variable Selection}
\usage{
kNNvs(
  train_x,
  test_x,
  cl_train,
  cl_test,
  k,
  model = c("regression", "classifiation")
)
}
\arguments{
\item{train_x}{matrix or data frame of training set}

\item{test_x}{matrix or data frame of test set}

\item{cl_train}{factor of true classifications of training set}

\item{cl_test}{factor of true classifications of test set}

\item{k}{the number of neighbors}

\item{model}{regression or classifiation}
}
\value{
ACC or MSE, best variable combination, estimate value yhat
}
\description{
k Nearest Neighbors with Grid Search Variable Selection
}
\details{
kNNvs is simply use add one and then compare acc to pick the best variable set for the knn model
}
\examples{
{
   data(iris3)
   train_x <- rbind(iris3[1:25,,1], iris3[1:25,,2], iris3[1:25,,3])
   test_x <- rbind(iris3[26:50,,1], iris3[26:50,,2], iris3[26:50,,3])
   cl_train<- cl_test<- factor(c(rep("s",25), rep("c",25), rep("v",25)))
   k<- 5
   # cl_test is not null
   mymodel<-kNNvs(train_x,test_x,cl_train,cl_test,k,model="classifiation")
   mymodel
   # cl_test is null
   mymodel<-kNNvs(train_x,test_x,cl_train,cl_test=NULL,k,model="classifiation")
   mymodel
   }
}
