EXPECTED_ESP_IDS = [str(i + 1) for i in range(4)] # ids esp


data_by_esp = {esp_id: [] for esp_id in EXPECTED_ESP_IDS}
print(data_by_esp)
processed_ids_in_parsing = set()
print(processed_ids_in_parsing)