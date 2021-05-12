#' k Nearest Neighbors with Grid Search Variable Selection
#' @param train_x matrix or data frame of training set
#' @param test_x  matrix or data frame of test set
#' @param cl_train factor of true classifications of training set
#' @param cl_test factor of true classifications of test set
#' @param k the number of neighbors
#' @param model regression or classifiation
#' @return ACC or MSE, best variable combination, estimate value yhat
#' @details kNNvs is simply use add one and then compare acc to pick the best variable set for the knn model
#' @examples {
#'    data(iris3)
#'    train_x <- rbind(iris3[1:25,,1], iris3[1:25,,2], iris3[1:25,,3])
#'    test_x <- rbind(iris3[26:50,,1], iris3[26:50,,2], iris3[26:50,,3])
#'    cl_train<- cl_test<- factor(c(rep("s",25), rep("c",25), rep("v",25)))
#'    k<- 5
#'    # cl_test is not null
#'    mymodel<-kNNvs(train_x,test_x,cl_train,cl_test,k,model="classifiation")
#'    mymodel
#'    # cl_test is null
#'    mymodel<-kNNvs(train_x,test_x,cl_train,cl_test=NULL,k,model="classifiation")
#'    mymodel
#'    }
#' @export



kNNvs<- function(train_x,test_x,cl_train,cl_test,k,model=c("regression","classifiation") )
{
 if(model=="regression"){
   if(!is.null(cl_test)){
     nnew     <- nrow(test_x)
     yhat.knn <- numeric(nnew)
     n_x<- ncol(test_x)
     y_hat_list<-as.list(NULL)
     mse_te_min<- rep(0,n_x)
     bag<- c(0:n_x)                              # the number of variable in the test data, since its add one so there will be n(test_x) round comparsion
     n_max<- 0                                   # the variable win around the peer model
     a <- 0                                      # save the variable information,the wining model's variabls
     outcome_list<- as.list(NA)                  # the wining model includs varibals
     for (h in 1:n_x) {
       bag<- bag[bag != n_max]
       mse_te<-rep(0,length(bag))
       a<-c(a,n_max)[c(a,n_max)!=0]
       y_hat_list_b<-as.list(NULL)
       for (j in 1:length(bag)) {
         b<- c(a,bag[j])
         test_xnew<- as.data.frame(test_x[,b])        # test_xnew is the test data used in this round
         train_xnew<-as.data.frame(train_x[,b])       # training data uded in this round

         for(i in 1:nnew){
           xn <- matrix(rep(t(test_xnew[i,]), times=nrow(train_xnew)), byrow=T, ncol=ncol(train_xnew))
           dist.x <- sqrt(rowSums((as.matrix(xn)-as.matrix(train_xnew))^2))
           s.dist <- sort(dist.x, index=T)
           neighbors <- s.dist$ix[1:k]
           w.x <- 1/(1+dist.x[neighbors])
           w <- w.x/sum(w.x)
           yhat.knn[i] <- sum(w*cl_train[neighbors])  #regression
         }
         y_hat_list_b[[j]] <- yhat.knn
         mse_te[j] <- sum((cl_test-yhat.knn)^2)/nnew
       }
       n_max<-bag[which(mse_te==min(mse_te))[1]]  # 1 means if there is tie i will take the first one
       mse_te_min[h]<-min(mse_te)
       list_te<-c(a,n_max)[c(a,n_max)!=0]
       outcome_list[[h]]<- c(list_te)
       y_hat_list[[h]]<- y_hat_list_b[[which(mse_te==min(mse_te))]]

     }

     return(list(mse=mse_te_min,outcome=outcome_list, bestmse = min(mse_te_min)[1], bestset = (outcome_list[mse_te_min == min(mse_te_min)][1]),y_hat= y_hat_list[which(mse_te_min==min(mse_te_min))[1]] ) )

   }
   else{
     ind <- sample(2, nrow(train_x), replace = T, prob = c(.7, .3))
     training_X <- train_x[ind==1,]
     test_X <- train_x[ind==2, ]
     training_Y <- cl_train[ind==1]
     test_Y <- cl_train[ind==2]

     nnew     <- nrow(as.data.frame(test_X))
     yhat.knn <- numeric(nrow(test_X))
     n_x<- ncol(test_X)
     mse_te_min<- rep(0,n_x)
     bag<- c(0:n_x)                              # the number of variable in the test data, since its add one so there will be n(test_X) round comparsion
     n_max<- 0                                   # the variable win around the peer model
     a <- 0                                      # save the variable information,the wining model's variabls
     outcome_list<- as.list(NA)                  # the wining model includs varibals
     for (h in 1:n_x) {
       bag<- bag[bag != n_max]
       mse_te<-rep(0,length(bag))
       a<-c(a,n_max)[c(a,n_max)!=0]

       for (j in 1:length(bag)) {
         b<- c(a,bag[j])
         test_Xnew<- as.data.frame(test_X[,b])        # test_Xnew is the test data used in this round
         training_Xnew<-as.data.frame(training_X[,b])       # training data uded in this round

         for(i in 1:nnew){
           xn <- matrix(rep(t(test_Xnew[i,]), times=nrow(training_Xnew)), byrow=T, ncol=ncol(training_Xnew))
           dist.x <- sqrt(rowSums((as.matrix(xn)-as.matrix(training_Xnew))^2))
           s.dist <- sort(dist.x, index=T)
           neighbors <- s.dist$ix[1:k]
           w.x <- 1/(1+dist.x[neighbors])
           w <- w.x/sum(w.x)
           yhat.knn[i] <- sum(w*training_Y[neighbors])  #regression
         }
         mse_te[j] <- sum((test_Y-yhat.knn)^2)/nnew
       }
       n_max<-bag[which(mse_te==min(mse_te))[1]]  # 1 means if there is tie i will take the first one
       mse_te_min[h]<-max(mse_te)
       list_te<-c(a,n_max)[c(a,n_max)!=0]
       outcome_list[[h]]<- c(list_te)
     }
     bestset = outcome_list[mse_te_min == min(mse_te_min)][1]

     x=as.data.frame(train_x[,as.numeric(bestset[[1]])])
     y=cl_train
     xnew=as.data.frame(test_x[,as.numeric(bestset[[1]])])
     nnew     <- nrow(as.data.frame(test_x))
     yhat.knn2 <- as.numeric(NULL)

     for(i in 1:nnew)
     {
       xn <- matrix(rep(t(xnew[i,]), times=nrow(x)), byrow=T, ncol=ncol(x))
       dist.x <- sqrt(rowSums((as.matrix(xn)-as.matrix(x))^2))
       s.dist <- sort(dist.x, index=T)
       neighbors <- s.dist$ix[1:k]

       # This is the equidistant weight. Default
       #w <- 1/k

       # This is the negative exponential weighting scheme, uncomment for use
       #w <- exp(-dist.x[neighbors])/sum(exp(-dist.x[neighbors]))

       #This is the inverse distance weighting scheme. Uncomment for use
       w.x <- 1/(1+dist.x[neighbors])
       w <- w.x/sum(w.x)

       yhat.knn2[i] <- sum(w*y[neighbors])
     }
     return(list(train_mse=mse_te_min,train_outcome=outcome_list, train_bestmse = min(mse_te_min)[1], train_bestset = (outcome_list[mse_te_min == min(mse_te_min)][1]) ,yhat=yhat.knn2) )

   }
 }
   if(model=="classifiation"){
     if(!is.null(cl_test)){
       nnew     <- nrow(test_x)
       yhat.knn <- numeric(nnew)
       n_x<- ncol(test_x)
       y_hat_list<-as.list(NULL)
       pcc_te_max<- rep(0,n_x)
       bag<- c(0:n_x)                              # the number of variable in the test data, since its add one so there will be n(test_x) round comparsion
       n_max<- 0                                   # the variable win around the peer model
       a <- 0                                      # save the variable information,the wining model's variabls
       outcome_list<- as.list(NA)                  # the wining model includs varibals
       for (h in 1:n_x) {
         bag<- bag[bag != n_max]
         pcc_te<-rep(0,length(bag))
         a<-c(a,n_max)[c(a,n_max)!=0]
         y_hat_list_b<-as.list(NULL)

         for (j in 1:length(bag)) {
           b<- c(a,bag[j])
           test_xnew<- as.data.frame(test_x[,b])        # test_xnew is the test data used in this round
           train_xnew<-as.data.frame(train_x[,b])       # training data uded in this round

           for(i in 1:nnew){
             xn <- matrix(rep(t(test_xnew[i,]), times=nrow(train_xnew)), byrow=T, ncol=ncol(train_xnew))
             dist.x <- sqrt(rowSums((as.matrix(xn)-as.matrix(train_xnew))^2))
             s.dist <- sort(dist.x, index=T)
             neighbors <- s.dist$ix[1:k]
             w.x <- 1/(1+dist.x[neighbors])
             w <- w.x/sum(w.x)
             # yhat.knn[i] <- sum(w*y[neighbors])  #regression
             yhat.knn[i] <- names(sort(-table(cl_train[neighbors])))[1]  # classfication
           }
           y_hat_list_b[[j]] <- yhat.knn
           pcc_te[j] <- sum(diag(table(cl_test,yhat.knn)))/nnew
         }
         n_max<-bag[which(pcc_te==max(pcc_te))[1]]  # 1 means if there is tie i will take the first one
         pcc_te_max[h]<-max(pcc_te)
         list_te<-c(a,n_max)[c(a,n_max)!=0]
         outcome_list[[h]]<- c(list_te)
         y_hat_list[[h]]<- y_hat_list_b[which(pcc_te==max(pcc_te))[1]]

       }

       return(list(acc=pcc_te_max,outcome=outcome_list, bestacc = max(pcc_te_max)[1], bestset = (outcome_list[pcc_te_max == max(pcc_te_max)][1]),y_hat= y_hat_list[which(pcc_te_max==max(pcc_te_max))[1]] ) )

     }
     else{
       ind <- sample(2, nrow(train_x), replace = T, prob = c(.7, .3))
       training_X <- train_x[ind==1,]
       test_X <- train_x[ind==2, ]
       training_Y <- cl_train[ind==1]
       test_Y <- cl_train[ind==2]

       nnew     <- nrow(as.data.frame(test_X))
       yhat.knn <- numeric(nrow(test_X))
       n_x<- ncol(test_X)
       pcc_te_max<- rep(0,n_x)
       bag<- c(0:n_x)                              # the number of variable in the test data, since its add one so there will be n(test_X) round comparsion
       n_max<- 0                                   # the variable win around the peer model
       a <- 0                                      # save the variable information,the wining model's variabls
       outcome_list<- as.list(NA)                  # the wining model includs varibals
       for (h in 1:n_x) {
         bag<- bag[bag != n_max]
         pcc_te<-rep(0,length(bag))
         a<-c(a,n_max)[c(a,n_max)!=0]

         for (j in 1:length(bag)) {
           b<- c(a,bag[j])
           test_Xnew<- as.data.frame(test_X[,b])        # test_Xnew is the test data used in this round
           training_Xnew<-as.data.frame(training_X[,b])       # training data uded in this round

           for(i in 1:nnew){
             xn <- matrix(rep(t(test_Xnew[i,]), times=nrow(training_Xnew)), byrow=T, ncol=ncol(training_Xnew))
             dist.x <- sqrt(rowSums((as.matrix(xn)-as.matrix(training_Xnew))^2))
             s.dist <- sort(dist.x, index=T)
             neighbors <- s.dist$ix[1:k]
             w.x <- 1/(1+dist.x[neighbors])
             w <- w.x/sum(w.x)
             # yhat.knn[i] <- sum(w*y[neighbors])  #regression
             yhat.knn[i] <- names(sort(-table(training_Y[neighbors])))[1]  # classfication
           }

           pcc_te[j] <- sum(diag(table(test_Y,yhat.knn)))/nnew
         }
         n_max<-bag[which(pcc_te==max(pcc_te))[1]]  # 1 means if there is tie i will take the first one
         pcc_te_max[h]<-max(pcc_te)
         list_te<-c(a,n_max)[c(a,n_max)!=0]
         outcome_list[[h]]<- c(list_te)
       }
       bestset = outcome_list[pcc_te_max == max(pcc_te_max)][1]

       x=as.data.frame(train_x[,as.numeric(bestset[[1]])])
       y=cl_train
       xnew=as.data.frame(test_x[,as.numeric(bestset[[1]])])
       nnew     <- nrow(as.data.frame(test_x))
       yhat.knn2 <- as.numeric(NULL)

       for(i in 1:nnew)
       {
         xn <- matrix(rep(t(xnew[i,]), times=nrow(x)), byrow=T, ncol=ncol(x))
         dist.x <- sqrt(rowSums((as.matrix(xn)-as.matrix(x))^2))
         s.dist <- sort(dist.x, index=T)
         neighbors <- s.dist$ix[1:k]

         # This is the equidistant weight. Default
         #w <- 1/k

         # This is the negative exponential weighting scheme, uncomment for use
         #w <- exp(-dist.x[neighbors])/sum(exp(-dist.x[neighbors]))

         #This is the inverse distance weighting scheme. Uncomment for use
         w.x <- 1/(1+dist.x[neighbors])
         w <- w.x/sum(w.x)

         #yhat.knn[i] <- sum(w*y[neighbors])
         yhat.knn2[i] <- names(sort(-table(y[neighbors])))[1]  # classfication
       }
       return(list(train_acc=pcc_te_max,train_outcome=outcome_list, train_bestacc = max(pcc_te_max)[1], train_bestset = (outcome_list[pcc_te_max == max(pcc_te_max)][1]) ,yhat=yhat.knn2) )

     }
}
}


