library(survival)
library(survminer)
library(survex)
library(randomForestSRC)
library(ggsurvfit)
library(ggplot2)
library(pec)
library(caret)
library(survivalsvm)
library(SurvMetrics)
library(gbm)
library(mlr3proba)
library(mlr3extralearners)
library(mlr3pipelines)
library(paradox)
library(mlr3tuning)
library(survivalmodels)

print('Starting...')


data_train <- read.csv("dataset//data_CPAP_train_new1.csv")
data_test <- read.csv("dataset//data_CPAP_test_new1.csv")

data_train <- data_train[, -c(4,24,26,27,28,29)] #drop other columns not considered in analysis
data_test <- data_test[, -c(4,24,26, 27,28,29)] #drop other columns not considered in analysis

EXPLAIN <- FALSE #set TRUE for computing explanation

data_train$Durata.follow.up.da.dimissione<-data_train$Durata.follow.up.da.dimissione/30
data_test$Durata.follow.up.da.dimissione<-data_test$Durata.follow.up.da.dimissione/30


time <- data_train$Durata.follow.up.da.dimissione
status <- data_train$Status

find_categorical_variables <- function(data) {
  categorical_vars <- character()  # Inizializza un vettore vuoto per le variabili categoriche

  for (col in names(data)) {
    if (is.factor(data[[col]]) || is.character(data[[col]])) {
      categorical_vars <- c(categorical_vars, col)
    } else if (is.numeric(data[[col]])) {
      unique_values <- unique(data[[col]])
      if (length(unique_values) < 10) {  # Arbitrarily consider variables with less than 10 unique values to be categorical

        categorical_vars <- c(categorical_vars, col)
      }
    }
  }

  return(categorical_vars)
}

trained_models <- list() #This vector will contain all the trained models

print('     Cox PH Model')
cox_model <- coxph(Surv(time,status) ~ ., data =  data_train[, -c(1,2)], x=TRUE, model = TRUE)
trained_models[[1]] <- cox_model
summary(cox_model)
cox_explainer_train <- survex::explain(cox_model,data =  data_train[, -c(1,2)],
                                   y = survival::Surv(data_train$Durata.follow.up.da.dimissione, data_train$Status),
                                   verbose=FALSE)
cox_explainer_test <- survex::explain(cox_model,data = data_test[,-c(1,2)],
                                 y = survival::Surv(data_test$Durata.follow.up.da.dimissione, data_test$Status),
                                 verbose=FALSE)
cox_perf_test <- model_performance(cox_explainer_test)
cox_perf_train <- model_performance(cox_explainer_train)

print('     Random Forest Model')
rf_formula <- Surv(Durata.follow.up.da.dimissione,Status) ~ .
rf_model <- rfsrc(rf_formula, data = data_train, ntree = 50, nsplit = 10)
trained_models[[2]] <- rf_model
rf_explainer_train <- survex::explain(rf_model,data =  data_train,
                               y = survival::Surv(data_train$Durata.follow.up.da.dimissione, data_train$Status),
                               verbose=FALSE)
rf_explainer_test <- survex::explain(rf_model,data =  data_test[,
                                 y = survival::Surv(data_test$Durata.follow.up.da.dimissione, data_test$Status),
                                 verbose=FALSE)
rf_perf_test <- model_performance(rf_explainer_test)
rf_perf_train <- model_performance(rf_explainer_train)

print('     GBM Model')
train_task <- as_task_surv(x = data_train,
                           time = "Durata.follow.up.da.dimissione",
                           event = "Status")
learner <- lrn("surv.gbm", n.trees=20, interaction.depth = 10, bag.fraction=0.9, shrinkage=0.1)
gbm_model <- as_learner(ppl(
"distrcompositor",
learner = learner,
estimator = "kaplan",
form = "ph"))

gbm_model$train(train_task)
class(gbm_model) <- c(class(gbm_model), "LearnerSurv")
trained_models[[3]] <- gbm_model

gbm_explainer_train <- explain(gbm_model,
                       data = data_train[, -c(1,2)],
                       y = Surv(data_train$Durata.follow.up.da.dimissione, data_train$Status),
                       label = "GBM model for Train",
                       verbose = FALSE)

gbm_explainer_test <- explain(gbm_model,
                            data = data_test[, -c(1,2)],
                            y = Surv(data_test$Durata.follow.up.da.dimissione, data_test$Status),
                            label = "GBM",
                            verbose = FALSE)
gbm_perf_test <- model_performance(gbm_explainer_test)
gbm_perf_train <- model_performance(gbm_explainer_train)

print('     SVM Model')
svm_model <- as_learner(ppl(
  "distrcompositor",
  learner = lrn("surv.svm", type = "vanbelle2", diff.meth = "makediff3", gamma.mu = 0.1),
  estimator = "kaplan",
  form = "ph"))
svm_model$train(train_task)

trained_models[[4]] <- svm_model

svm_predict <- function(model, newdata, times)
{
  if (nrow(newdata) == 1){
      newdata <- rbind(newdata, newdata)
      t(model$predict_newdata(newdata)$distr$survival(times))[1, , drop=FALSE]
  }
  else{
      t(model$predict_newdata(newdata)$distr$survival(times))
  }
}

svm_explainer_train <- explain(svm_model,
                           data = data_train[, -c(1,2)],
                           y = Surv(data_train$Durata.follow.up.da.dimissione, data_train$Status),
                           predict_survival_function = svm_predict,
                           label = "SVM model for Train",
                           verbose = FALSE)
svm_explainer_test <- explain(svm_model,
                           data = data_test[, -c(1,2)],
                           y = Surv(data_test$Durata.follow.up.da.dimissione, data_test$Status),
                           predict_survival_function = svm_predict,
                           label = "SVM",
                           verbose = FALSE)
svm_perf_test <- model_performance(svm_explainer_test)
svm_perf_train <- model_performance(svm_explainer_train)

plot(cox_explainer_train,rf_explainer_train,gbm_explainer_train,
     svm_explainer_train, metrics_type = "scalar")
plot(cox_explainer_train,rf_explainer_train,gbm_explainer_train,
     svm_explainer_train, metrics_type = "scalar")


best_model <- trained_models[[1]] #substitute with best model according performance

if(EXPLAIN){
	best_explainer_train <- survex::explain(best_model,
                           data = data_train[, -c(1,2)],
                           y = Surv(data_train$Durata.follow.up.da.dimissione, data_train$Status),
                           label = paste(best_model$id, "test_exp", sep = " ")
  )

    best_explainer_test <- survex::explain(best_model,
                           data = data_test[, -c(1,2)],
                           y = Surv(data_test$Durata.follow.up.da.dimissione, data_test$Status),
                           label = paste(best_model$id, " test explainer", sep = " ")
  )
  m_parts  <- model_parts(best_explainer_train, new_observation = data_test[, -c(1, 2)]) #variable importance for model
  model_prof_train <- model_profile(best_explainer_train, categorical_variables = find_categorical_variables(data_train[, -c(1, 2)]))
  model_prof_test <- model_profile(best_explainer_test, categorical_variables = find_categorical_variables(data_test[, -c(1, 2)]))
  plot(m_parts, max_vars = 7)
  plot(model_prof_train)
  plot(model_prof_test)

  start_time <- Sys.time()
  global_survshap <- predict_parts(best_explainer,
                              new_observation = data_test[, -c(1, 2)],
                              type = "survshap",
                              aggregation_method = "integral" )
  end_time <- Sys.time()

  time_diff <- (end_time-start_time)
  cat( "Elapsed Time for global survshap", time_diff, sep = " ")
  plot(global_survshap, max_vars = 7)
}