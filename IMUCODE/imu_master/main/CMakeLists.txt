idf_component_register(
    SRCS "main.c" "esp32_s3_szp.c"
    INCLUDE_DIRS "."
    REQUIRES
        driver
        esp_driver_gpio
        esp_driver_spi
        esp_driver_ledc
        esp_lcd
        esp_wifi
        esp_event
        nvs_flash
        esp_netif
        esp_timer
)
