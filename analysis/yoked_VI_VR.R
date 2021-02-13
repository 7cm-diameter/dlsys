library(tidyverse)
library(ggcolors)

simulation_result <- read.csv("./data/yoked_VI_VR.csv")

grouped_result <- simulation_result %>%
  split(., list(.$condition, .$session), drop = T) %>%
  lapply(., function(d) {
    mean_gk <- mean(d$gk)
    mean_hk <- mean(d$hk)
    sd_gk <- sd(d$gk)
    sd_hk <- sd(d$hk)
    data.frame(condition = unique(d$condition),
               session = unique(d$session),
               mean_gk = mean_gk,
               mean_hk = mean_hk)
}) %>%
  do.call(rbind, .)

colors <- c("gk" = "#e27878", "hk" = "#84a0c6", "gk + hk" = "#a093c7")

ggplot(data = grouped_result) +
  geom_line(aes(x = session,
                y = mean_gk,
                color = "gk"),
            size = 1.5) +
  geom_line(aes(x = session,
                y = mean_hk,
                color = "hk"),
            size = 1.5) +
  geom_line(aes(x = session,
                y = mean_gk + mean_hk,
                color = "gk + hk"),
            size = 1.5) +
  labs(x = "session",
       y = "response strength") +
  scale_color_manual(values = colors) +
  facet_wrap(~condition) +
  theme_iceberg_dark()
