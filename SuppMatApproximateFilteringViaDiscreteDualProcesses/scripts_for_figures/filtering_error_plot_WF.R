library(tidyverse)

to_plot = read_csv("saves_for_figures/filtering_error_WF.csv")

cbPalette <- c("#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00")


to_plot2 = to_plot %>%
    as_tibble %>%
    (function(df){
        df %>%
            select(-contains('CI')) %>%
            gather(variable, values, contains('expected')) %>%
            mutate(variable = gsub('expected_', '', variable)) %>%
            mutate(variable = gsub('_error', '', variable)) %>%
            left_join(        
                df %>%
                    select(-contains('expected')) %>%
                    gather(variable, error, contains('CI')) %>%
                    mutate(variable = gsub('_CI_size', '', variable))
                )
    }) %>%
    mutate(variable = variable %>% 
                        gsub("first_moment", "First moment", .) %>%
                        gsub("second_moment", "Second moment", .) %>%
                        gsub("signal_retrieval", "Signal retrieval", .) %>%
                        gsub("sd", "Standard Deviation", .) 
                        ) %>%
    mutate(method = method %>% 
                        gsub("BD_Moran_Gillespie", "BD Gillespie Moran", .) %>%
                        gsub("BD_WF_Gillespie", "BD Gillespie WF", .) %>%
                        gsub("BD_WF_diffusion", "BD diffusion WF", .) %>%
                        gsub("PD_integrated", "PD", .) %>%
                        gsub("PF", "Bootstrap PF", .) 
                        ) %>%
    filter(method != "PD_Gillespie") %>%
    mutate(method = factor(method, levels = c("PD", "BD Gillespie Moran", "BD Gillespie WF", "BD diffusion WF", "Bootstrap PF" )))


pl2 = to_plot2 %>%
    filter(time_step == 0.01) %>%
    filter(method != "Pruning (PD)") %>%
    filter(method != "Exact") %>%
    filter(variable != "Second moment") %>%
    ggplot(aes(x = Nparts, y = values, colour = method, group = method)) + 
    theme_bw() + 
    facet_wrap(~variable, scale = "free_y", nrow = 1) +
    geom_point() + 
    geom_line(linewidth = 0.25) +
    geom_errorbar(aes(ymin = values - error, ymax = values + error, group = interaction(method, Nparts)), linewidth = 0.75) + 
    scale_colour_manual(values=cbPalette, name = "") +
    ylab("Error and CI") + 
    xlab("# particles")


    ggsave("figures/filtering_error_WF_2.pdf",  pl2, height = 3, width = 10)

