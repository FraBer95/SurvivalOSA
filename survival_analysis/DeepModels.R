library(mlr3extralearners)
library(survex)
library(randomForestSRC)
library(ggsurvfit)
library(ggplot2)
library(pec)
library(caret)
library(survivalsvm)
library(SurvMetrics)
library(mlr3proba)
library(mlr3pipelines)
library(paradox)
library(mlr3tuning)
library(survivalmodels)
library(reticulate)

print('Starting...')

#columns (1,2) = (time, event)

data_train <- read.csv("dataset//data_CPAP_train_new1.csv")
data_test <- read.csv("dataset//data_CPAP_test_new1.csv")

data_train <- data_train[, -c(4,24,26,27,28,29)] #drop other columns not considered in analysis
data_test <- data_test[, -c(4,24,26, 27,28,29)] #drop other columns not considered in analysis

EXPLAIN <- FALSE #set TRUE for computing explanation

data_train$Durata.follow.up.da.dimissione<-data_train$Durata.follow.up.da.dimissione/30
data_test$Durata.follow.up.da.dimissione<-data_test$Durata.follow.up.da.dimissione/30


time <- data_train$Durata.follow.up.da.dimissione
status <- data_train$Status



train_task <- as_task_surv(x = data_train,
                           time = "Durata.follow.up.da.dimissione",
                           event = "Status")

learners <- lrns(
 paste0("surv.", c("coxtime", "deephit" , "deepsurv", "loghaz", "pchazard")),
 frac = 0.3, early_stopping = TRUE, optimizer = "adam", epochs=100
)

find_categorical_variables <- function(data) {
  categorical_vars <- character()  # Inizializza un vettore vuoto per le variabili categoriche

  for (col in names(data)) {
    if (is.factor(data[[col]]) || is.character(data[[col]])) {
      categorical_vars <- c(categorical_vars, col)
    } else if (is.numeric(data[[col]])) {
      unique_values <- unique(data[[col]])
      if (length(unique_values) < 10) {  # Arbitrariamente, consideriamo le variabili con meno di 10 valori unici come categoriche
        categorical_vars <- c(categorical_vars, col)
      }
    }
  }

  return(categorical_vars)
}




model_performances <- list()
model_performances_tr <- list()
trained_models <- list()
for (learner in learners) {

  model_id <- learner$id
  print(cat("Training model: ", model_id))


    composite_learner <-  as_learner(ppl(
      "distrcompositor",
      learner = learner,
      estimator = "kaplan",
      form = "ph"
  ))

  composite_learner$train(train_task)

  class(composite_learner) <- c(class(composite_learner), "LearnerSurv")

  trained_models[[model_id]] <- composite_learner

  explainer_train<- survex::explain(composite_learner,
                           data = data_train[, -c(1,2)],
                           y = Surv(data_train$Durata.follow.up.da.dimissione, data_train$Status),
                           label = paste(model_id, " test explainer", sep = " ")
  )

    explainer_test <- survex::explain(composite_learner,
                           data = data_test[, -c(1,2)],
                           y = Surv(data_test$Durata.follow.up.da.dimissione, data_test$Status),
                           label = paste(model_id, " test explainer", sep = " ")
  )

  cat("Computing_performance for ", model_id )
  model_perf <- model_performance(explainer_test)
  model_perf_tr <- model_performance(explainer_train)

  model_performances[[model_id]] <- model_perf
  model_performances_tr[[model_id]] <- model_perf_tr


}



plot(model_performances_tr[[1]], model_performances_tr[[2]], model_performances_tr[[3]], model_performances_tr[[4]],
     model_performances_tr[[5]], metrics_type = "scalar")

plot(model_performances[[1]], model_performances[[2]], model_performances[[3]], model_performances[[4]],
     model_performances[[5]], metrics_type = "scalar")

best_model <- trained_models[[4]] #substitute with best model according performance

if (EXPLAIN){

    best_explainer <- survex::explain(trained_models[[3]],
                           data = data_train[, -c(1,2)],
                           y = Surv(data_train$Durata.follow.up.da.dimissione, data_train$Status),
                           label = paste(trained_models[[3]]$id, "test_exp", sep = " ")
  )

    best_explainer_test <- survex::explain(trained_models[[3]],
                           data = data_test[, -c(1,2)],
                           y = Surv(data_test$Durata.follow.up.da.dimissione, data_test$Status),
                           label = paste(trained_models[[3]]$id, " test explainer", sep = " ")
  )

  m_parts  <- model_parts(best_explainer_train, new_observation = data_test[, -c(1, 2)]) #variable importance for model
  model_prof_tr <- model_profile(best_explainer_train, categorical_variables = find_categorical_variables(data_train[, -c(1, 2)]))
  model_prof_test <- model_profile(best_explainer_test, categorical_variables = find_categorical_variables(data_test[, -c(1, 2)]))

  plot(global_survshap, max_vars = 7)
  plot(model_prof_tr)
  plot(model_prof_test)

  start_time <- Sys.time()
  global_survshap <- predict_parts(best_explainer,
                              new_observation = data_test[, -c(1, 2)],
                              type = "survshap",
                              aggregation_method = "integral" )
  end_time <- Sys.time()

  time_diff <- (end_time-start_time)
  cat( "Elapsed Time for global survshap", time_diff, sep = " ")

  #save.image("./Rdata/WS/deepWS_XAI.Rdata")

  plot(pred_parts)
  plot(pred_parts, geom_type = "beeswarm")


}