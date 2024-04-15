library(survival)
library(survminer)

#Plot Survival curves

data <- read.csv("data_processed.csv")
columns <- list("Status","Sesso","Asthma", "CAD", "Fibrillazione.atriale", "Valvolopatia", "Heart.failure", "Renal.dysfunction",
                "Cardiopatia.dilatativa.idopatica", "GOLD", "Previous.CV.events", "Diabete", "Ipertensione", "Malignancy")

data$Durata.follow.up.da.dimissione<-data$Durata.follow.up.da.dimissione/30

for (i in seq_along(columns)){
  surv_model <- survfit(Surv(data$Durata.follow.up.da.dimissione, data$Status) ~ data[i], data = data)

  summary(surv_model)

  # Calcola i p-value
  p_value <- survdiff(Surv(data$Durata.follow.up.da.dimissione, data$Status) ~ data[i], data = data)
  p_value

  # Plotta le curve di sopravvivenza
  ggsurvplot(surv_model, data = data, risk.table = TRUE, pval = TRUE)
}


for (feature in columns){
  formula_surv <- Surv(data$Durata.follow.up.da.dimissione, data$Status) ~ data[, feature]

  s <- survfit(formula_surv, data = columns) |>
    ggsurvfit(linewidth = 1) +
    add_confidence_interval() +
    add_risktable() +
    add_pvalue()
  plot(s)
}