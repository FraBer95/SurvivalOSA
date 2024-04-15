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
data_train <- read.csv("dataset//data_CPAP_train.csv")
data_test <- read.csv("dataset//data_CPAP_test.csv")

data_train <- lapply(data_train, as.numeric)
data_train <- as.data.frame(data_train)
data_test <- lapply(data_test, as.numeric)
data_test <- as.data.frame(data_test)

data_train$Durata.follow.up.da.dimissione<-data_train$Durata.follow.up.da.dimissione/30
data_test$Durata.follow.up.da.dimissione<-data_test$Durata.follow.up.da.dimissione/30

data_train <- data_train[, -c(4,24,26,27,28,29)]
data_test <- data_test[, -c(4,24,26, 27,28,29)]






cox_train <- TRUE
expl_cox <- TRUE

rf_train <- FALSE
expl_rf <- FALSE

gbm_train <- FALSE
expl_gbm <- FALSE

svm_train <- FALSE
expl_svm <- FALSE

time <- data_train$Durata.follow.up.da.dimissione
status <- data_train$Status


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

plot_sc <- FALSE
if (plot_sc){
  not_col <- c('AHI', 'SaO2.min', 'Years_of_CPAP', 'CPAP_0_5','CPAP_5_10', 'CPAP.10')
  columns <- names(data)
  temp <- data[,!columns %in% not_col]
  columns <- names(temp)
  time <- data$Durata.follow.up.da.dimissione
  status <- data$Status
  print("Plotting Survival Curves...")
  for (feature in columns[3:length(columns)]){
  fit  <- survfit(Surv(time,status) ~ data[[feature]])
  n_val <- length(unique(data[[feature]]))
  values <- unique(data[[feature]])
  labs <- character(0)
  for(i in 1:n_val){
    string <- paste(feature," = ", i-1)
    labs[i] <- string
  }
  print(feature)
  p <- ggsurvplot(
    fit,
    data = data,
    risk.table = TRUE,
    pval = TRUE,
    main = paste("Curva di sopravvivenza per", feature),
    conf.int = TRUE,
    legend.labs = labs,
    legend.title = feature)

  print(p)
  # combined_plot <- grid.arrange(
  #   p$plot,
  #   p$table,
  #   nrow = 2,
  #   heights = c(3, 1))
  # feature = gsub("\\.", "_", feature)
  # feature = tolower(feature)
  # ggsave(paste0("images//survival_curves//",feature, "_surv_curv.png"), plot = combined_plot, width = 10, height = 10)
  # print(0)

}
}

if (cox_train){
  print("Training COX Model...")
  cox_model <- coxph(Surv(time,status) ~ ., data =  data_train[, -c(1,2)], x=TRUE, model = TRUE)
  cox_summary <- summary(cox_model)
  cox_explainer <- survex::explain(cox_model,data =  data_train[, -c(1,2)],
                                   y = survival::Surv(data_train$Durata.follow.up.da.dimissione, data_train$Status),
                                   verbose=FALSE)
  cox_explainer_test <- survex::explain(cox_model,data = data_test[,-c(1,2)],
                                 y = survival::Surv(data_test$Durata.follow.up.da.dimissione, data_test$Status),
                                 verbose=FALSE)


  if (expl_cox){
    print("---> Explenability COX Model...")
    modelparts_cox <- model_parts(cox_explainer_test, output_type="survival")
    plot(modelparts_cox)

    #save.image("Rdata\\modelparts_COX.Rdata")
    # survlime_cox <- matrix(0,nrow=1, ncol=dim(data_test[, -c(1,2,4,24,26,27,28,29)])[2])
    # for(p in 1:dim(data_test[, -c(1,2,4,24,26,27,28,29)])[1]){
    #   lime <- predict_parts(cox_explainer,new_observation = data_test[, -c(1,2,4,24,26,27,28,29)][p,], type = "survlime",
    #                         categorical_variable = find_categorical_variables(data_test[, -c(1,2,4,24,26,27,28,29)]))
    #   result <- as.matrix(lime$result)
    #   for(j in 1:dim(result)[2]){
    #     survlime_cox[1,j] <- survlime_cox[1,j]+result[1,j]
    #   }
    #   print(p)
    # }
    # barplot(survlime_cox, names.arg = colnames(data_test[, -c(1,2,4,24,26,27,28,29)]),
    #         xlab = "SurvLIME", ylab = "Features")
  }

  print("Cox Terminated!")
}

if(rf_train){
  print("Training SURVIVAL RANDOM FOREST Model ...")
  formula <- Surv(Durata.follow.up.da.dimissione,Status) ~ .
  model_rf <- rfsrc(formula, data = data_train[, -c(4,24,26,27,28,29)], ntree = 50, nsplit = 10)
  rf_explainer <- survex::explain(model_rf,data =  data_train[, -c(4,24,26,27,28,29)],
                                   y = survival::Surv(data_train$Durata.follow.up.da.dimissione, data_train$Status),
                                   verbose=FALSE)
  rf_explainer_test <- survex::explain(model_rf,data =  data_test[,-c(4,24,26,27,28,29)],
                                 y = survival::Surv(data_test$Durata.follow.up.da.dimissione, data_test$Status),
                                 verbose=FALSE)
  if(expl_rf){
    print("---> Explenability SRF Model...")
    modelparts_rf <- model_parts(rf_explainer_test,type = "variable_importance", output_type="survival")
    save.image("Rdata\\modelparts_RF.Rdata")
    # data_rf_test$Status <- as.integer(data_rf_test$Status)
    # survlime_rf <- matrix(0,nrow=1, ncol=dim(data_test[, -c(4,24,26,27,28,29)])[2])
    # for(p in 1:dim(data_test[, -c(4,24,26,27,28,29)])[1]){
    #   lime <- predict_parts(rf_explainer, new_observation = data_test[, -c(4,24,26,27,28,29)][p,], type = "survlime",
    #                         categorical_variable = find_categorical_variables(data_test[, -c(4,24,26,27,28,29)]))
    #   result <- as.matrix(lime$result)
    #   for(j in 1:dim(result)[2]){
    #     survlime_rf[1,j] <- survlime_rf[1,j]+result[1,j]
    #   }
    #   print(p)
    # }
    # barplot(survlime_rf, names.arg = colnames(data_test[, -c(4,24,26,27,28,29)]),
    #         xlab = "SurvLIME", ylab = "Features")
  }
  print("SRF Terminated!")
}

if(gbm_train){
  print("Training GBM Model...")
  train_task <- as_task_surv(x = data_train[, -c(4,24,26,27,28,29)],
                           time = "Durata.follow.up.da.dimissione",
                           event = "Status")
  learner <- lrn("surv.gbm", n.trees=20, interaction.depth = 10, bag.fraction=0.9, shrinkage=0.1)
  gbm_composite_learner <- as_learner(ppl(
    "distrcompositor",
    learner = learner,
    estimator = "kaplan",
    form = "ph"))

  gbm_composite_learner$train(train_task)
  class(gbm_composite_learner) <- c(class(gbm_composite_learner), "LearnerSurv")

  gbm_explainer <- explain(gbm_composite_learner,
                           data = data_train[, -c(1,2,4,24,26,27,28,29)],
                           y = Surv(data_train$Durata.follow.up.da.dimissione, data_train$Status),
                           label = "GBM model for Train",
                           verbose = FALSE)

  gbm_explainer_test <- explain(gbm_composite_learner,
                                data = data_test[, -c(1,2,4,24,26,27,28,29)],
                                y = Surv(data_test$Durata.follow.up.da.dimissione, data_test$Status),
                                label = "GBM",
                                verbose = FALSE)
  if(expl_gbm){
    print("---> Explenability GBM Model...")
    modelparts_gbm <- model_parts(gbm_explainer_test,type = "variable_importance", output_type="survival")
    save.image("Rdata\\modelparts_GBM.Rdata")
  }
  print("GBM Terminated!")

}

if(svm_train){
    print("Training SVM Model...")
    svm_composite_learner <- as_learner(ppl(
      "distrcompositor",
      learner = lrn("surv.svm", type = "vanbelle2", diff.meth = "makediff3", gamma.mu = 0.1),
      estimator = "kaplan",
      form = "ph"))

  ## Train the gbm model
  svm_composite_learner$train(train_task)

  ## Create the explainer, adding `LearnerSurv`to the class is
  ## necessary for automatic explainer creation
  class(svm_composite_learner) <- c(class(svm_composite_learner), "LearnerSurv")


  ### SVM does not seem to want to predict when there is only one observation
  ### I manually provide a custom function as a workaround
  svm_predict <- function(model, newdata, times) {
      if (nrow(newdata) == 1){
          newdata <- rbind(newdata, newdata)
          t(model$predict_newdata(newdata)$distr$survival(times))[1, , drop=FALSE]
      }
      else{
          t(model$predict_newdata(newdata)$distr$survival(times))
      }

  }

  ## Create the svm explainer
  svm_explainer <- explain(svm_composite_learner,
                           data = data_train[, -c(1,2,4,24,26,27,28,29)],
                           y = Surv(data_train$Durata.follow.up.da.dimissione, data_train$Status),
                           predict_survival_function = svm_predict,
                           label = "SVM model for Train",
                           verbose = FALSE)

  svm_explainer_test <- explain(svm_composite_learner,
                                data = data_test[, -c(1,2,4,24,26,27,28,29)],
                                y = Surv(data_test$Durata.follow.up.da.dimissione, data_test$Status),
                                predict_survival_function = svm_predict,
                                label = "SVM",
                                verbose = FALSE)
  if(expl_svm){
    print("---> Explenability SVM Model...")
    modelparts_svm <- model_parts(svm_explainer_test,type = "variable_importance", output_type="survival")
    save.image("Rdata\\modelparts_SVM.Rdata")

  }
  print("SVM Terminated!")
}

# Evaluation Metrics
print("Evaluating Models... ")

cox_perf <- model_performance(cox_explainer_test)
rf_perf <- model_performance(rf_explainer_test, new_observation = data_test[, -c(1,2,4,24,26,27,28,29)])
gbm_perf <- model_performance(gbm_explainer_test, new_observation = data_test[, -c(1,2,4,24,26,27,28,29)])
svm_perf <- model_performance(svm_explainer_test, new_observation = data_test[, -c(1,2,4,24,26,27,28,29)])

if(FALSE){
  deepsurv <- explain(svm_composite_learner,
                      data = data_test[, -c(1,2)],
                      y = Surv(data_test$Durata.follow.up.da.dimissione, data_test$Status),
                      predict_survival_function = svm_predict,
                      label = "DeepSurv",
                      verbose = FALSE)
  deephit <- explain(svm_composite_learner,
                     data = data_test[, -c(1,2)],
                     y = Surv(data_test$Durata.follow.up.da.dimissione, data_test$Status),
                     predict_survival_function = svm_predict,
                     label = "DeepHit",
                     verbose = FALSE)
  deepsurv_perf <- model_performance(deepsurv, new_observation = data_test[, -c(1,2,4,24,26,27,28,29)])
  deephit_perf <- model_performance(deephit, new_observation = data_test[, -c(1,2,4,24,26,27,28,29)])
  deepsurv_perf$result$'C-index'[1]<- 0.784641
  deepsurv_perf$result$'Integrated Brie score'[1]<- 0.1220
  deephit_perf$result$'C-index'[1]<- 0.748943
  deephit_perf$result$'Integrated Brier score'[1] <- 0.1500

  plot(cox_perf,rf_perf,
       gbm_perf,svm_perf,
       deepsurv_perf, deephit_perf, metrics_type = "scalar")
}

plot(cox_perf,rf_perf, gbm_perf,svm_perf,metrics_type = "scalar")
save.image("Rdata\\workspace_analysis.Rdata")