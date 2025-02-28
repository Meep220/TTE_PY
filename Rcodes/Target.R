# Set up for ITT and Perprotocol of Target Trial Emulation
library(TrialEmulation)

#R is a loosely typed Language meaning you can assign variables to anything...

trial_pp <- trial_sequence(estimand = "PP") #This is how Assignments in R are made through Variable <- function or assignment

trial_ITT <- trial_sequence(estimand = "ITT")

#Creates Temporary directory files for the trial
trial_pp_dir <- file.path(tempdir(),"trial_pp")
dir.create(trial_pp_dir)

trial_ITT_dir <- file.path(tempdir(),"trial_ITT")
dir.create(trial_ITT_dir)

#Data Setup
data("data_censored")
head(data_censored)

#Per-Protocol Portion
#It looks similar to a data structure/structure in C
trial_pp <- trial_pp |>
    set_data(
        data = data_censored, #basically what data you are using, then the following are the properties of the data
        id = "id",
        period = "period",
        treatment = "treatment",
        outcome = "outcome",
        eligible = "eligible"
    )
# Another way in using setData without the |> pipe use the variable first in the set data then followed by the data and properties.
trial_ITT <- set_data(
        trial_ITT,
        data = data_censored,
        id = "id",
        period = "period",
        treatment = "treatment",
        outcome = "outcome",
        eligible = "eligible"
    )

# Switch weight model

trial_pp <- set_switch_weight_model(
        trial_pp,
        numerator = ~ age,
        denominator = ~ age + x1 + x3,
        model_fitter = stats_glm_logit(save_path = file.path(trial_pp_dir,"switch_models"))
    )
# using @ will display its specific function or property
trial_pp@switch_weights #displays the function above and what parameters it is using for weights

#Censor Weight model

trial_pp <- set_censor_weight_model(
    trial_pp,
    censor_event = "censored",
    numerator = ~ x2,
    denominator = ~ x2 + x1,
    pool_models = "none",
    model_fitter = stats_glm_logit(save_path = file.path(trial_pp_dir,"switch_models"))
    
)


trial_ITT <- set_censor_weight_model(
    trial_ITT,
    censor_event = "censored",
    numerator = ~ x2,
    denominator = ~ x2 + x1,
    pool_models = "numerator",
    model_fitter = stats_glm_logit(save_path = file.path(trial_pp_dir,"switch_models"))
)



# Weight calculation

trial_pp <- calculate_weights(trial_pp)
trial_ITT <- calculate_weights(trial_ITT)
# This shows the weight models in the trial, the parameters and everything.
show_weight_models(trial_ITT)

#Specifying Outcome Models

trial_pp <- set_outcome_model(trial_pp)
trial_ITT <- set_outcome_model(trial_ITT, adjustment_terms = ~x2)

trial_pp <- set_expansion_options(
  trial_pp,
  output = save_to_datatable(),
  chunk_size = 500 # the number of patients to include in each expansion iteration
)

trial_ITT <- set_expansion_options(
  trial_ITT,
  output = save_to_datatable(),
  chunk_size = 500
)

trial_pp  <- expand_trials(trial_pp)
trial_ITT <- expand_trials(trial_ITT)



trial_ITT <- load_expanded_data(trial_ITT, seed = 1234, p_control = 0.5)

trial_ITT <- fit_msm(
  trial_ITT,
  weight_cols    = c("weight", "sample_weight"),
  modify_weights = function(w) { # winsorization of extreme weights
    q99 <- quantile(w, probs = 0.99)
    pmin(w, q99)
  }
)

trial_ITT

preds <- predict(
  trial_ITT,
  newdata       = outcome_data(trial_ITT)[trial_period == 1, ],
  predict_times = 0:10,
  type          = "survival",
)

plot(preds$difference$followup_time, preds$difference$survival_diff,
  type = "l", xlab = "Follow up", ylab = "Survival difference")
lines(preds$difference$followup_time, preds$difference$`2.5%`, type = "l", col = "red", lty = 2)
lines(preds$difference$followup_time, preds$difference$`97.5%`, type = "l", col = "red", lty = 2)

