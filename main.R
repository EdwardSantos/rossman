library(data.table)
library(reshape2)
library(gbm)

dt_train = fread("Data/train.csv")
dt_store = fread("Data/store.csv")

dt = merge(dt_train,dt_store, by="Store")

dt[StateHoliday=="0",state_holiday:=0]
dt[StateHoliday=="1",state_holiday:=1]
dt[SchoolHoliday=="0",school_holiday:=0]
dt[SchoolHoliday=="1",school_holiday:=1]
dt[,StoreType:=as.factor(StoreType)]
dt[,Assortment:=as.factor(Assortment)]
dt[,PromoInterval:=as.factor(PromoInterval)]

pred_names = names(dt)
pred_names = pred_names[!(pred_names %in% c("Sales","Date","StateHoliday","SchoolHoliday") ) ]

fit_formula = formula(paste0("Sales~", paste(pred_names,collapse = "+")))

# cluster on Dates.

gbm1 <- gbm(fit_formula,         # formula
            data=dt,                   # dataset
            distribution="gaussian",     # see the help for other choices
            n.trees=1000,                # number of trees
            shrinkage=0.01,              # shrinkage or learning rate,
            # 0.001 to 0.1 usually work
            interaction.depth=10,         # 1: additive model, 2: two-way interactions, etc.
            bag.fraction = 0.5,          # subsampling fraction, 0.5 is probably best
            train.fraction = 0.9,        # fraction of data for training,
            # first train.fraction*N used for training
            n.minobsinnode = 10,         # minimum total weight needed in each node
            cv.folds = 5,                # do 3-fold cross-validation
            keep.data=FALSE,              # keep a copy of the dataset with the object
            verbose=TRUE,               # don't print out progress
            n.cores=1)                   # use only a single core (detecting #cores is
            # error-prone, so avoided here)

# check performance using an out-of-bag estimator
# OOB underestimates the optimal number of iterations
best.iter <- gbm.perf(gbm1,method="cv")
print(best.iter)

# Generate prediction.
dt_test = fread("Data/test.csv")
dt_test = merge(dt_test,dt_store, by="Store")

dt_test[StateHoliday =="0",state_holiday :=0]
dt_test[StateHoliday =="1",state_holiday :=1]
dt_test[SchoolHoliday=="0",school_holiday:=0]
dt_test[SchoolHoliday=="1",school_holiday:=1]
dt_test[,StoreType :=as.factor(StoreType)]
dt_test[,Assortment:=as.factor(Assortment)]
dt_test[,PromoInterval:=as.factor(PromoInterval)]

prediction <- predict(gbm1, dt_test, best.iter)

# Id, Sales
output = data.table(Id=seq(1,length(prediction)),
                    Sales=prediction)

write.csv(output, 
          file="submission.csv", 
          sep = ",", 
          row.names = FALSE, 
          col.names = TRUE)



