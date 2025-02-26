# Set up for ITT and Perprotocol of Target Trial Emulation
library(TrialEmulation)


trial_pp <- trial_sequence(estimand = "PP")

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

trial_pp <- set_switch_weight_model(
        trial_pp,
        numerator = ~ age,
        denominator = ~ age + x1 + x3,
        model_fitter = stats_glm_logit(save_path = file.path(trial_pp_dir,"switch_models"))
    )
trial_pp@switch_weights





