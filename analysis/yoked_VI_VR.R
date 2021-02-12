library(tidyverse)


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
  theme(plot.background = element_rect(fill = "#161821"),
        panel.background = element_rect(fill = "#161821"),
        panel.grid.major = element_line(color = "#d2d4de"),
        panel.grid.minor = element_line(color = "#d2d4de"),
        panel.border = element_rect(color = "#d2d4de", fill = NA),
        legend.background = element_rect(color = "#6b7089", fill = "#161821"),
        legend.key = element_rect(fill = "#d2d4de"),
        legend.title = element_text(color = "#d2d4de"),
        legend.text = element_text(color = "#d2d4de"),
        strip.background = element_rect(color = "#6b7089", fill = "#6b7089"),
        axis.text = element_text(color = "#d2d4de"),
        text = element_text(color = "#d2d4de")
  )
