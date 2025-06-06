library(tidyverse)

to_plot = read_csv("saves_for_figures/filtering_error_varying_sigma.csv")

cbPalette <- c("#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7")

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
                        )

pl2 = to_plot2 %>%
    filter(sigma == 1) %>%
    filter(method != "Pruning (PD)") %>%
    filter(variable != "Second moment") %>%
    ggplot(aes(x = Nparts, y = values, colour = method, group = method)) + 
    theme_bw() + 
    facet_wrap(~variable, scale = "free_y", nrow = 1) +
    geom_point() + 
    geom_line(linewidth = 0.25) +
    geom_errorbar(aes(ymin = values - error, ymax = values + error, group = interaction(method, Nparts)), linewidth = 0.75) + 
    scale_colour_manual(values=cbPalette, name = "", label = c("BD", "PD", "BPF")) +
    ylab("Error and CI") + 
    xlab("# particles")

ggsave("figures/filtering_error_comparison_CIR_2.pdf",  pl2, height = 3, width = 10)
