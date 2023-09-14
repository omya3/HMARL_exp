slice_req_params = {
    "C1": {
        "NC": 4000,
        "NB": 1100,
        "NM": 2000,
        "arr_rate_mean": 7,
        "mean_bid_val": 25,
        "var": 1,
        "mean_duration": 90,
        "SLAV": 0,
        "rej_penalty": 100,
        "slav_penalty": 25,
    },
    "C2": {
        "NC": 2000,
        "NB": 500,
        "NM": 1500,
        "arr_rate_mean": 9,
        "mean_bid_val": 15,
        "var": 1.5,
        "mean_duration": 70,
        "SLAV": 5,
        "rej_penalty": 0,
        "slav_penalty": 15,
    },
    "C3": {
        "NC": 1500,
        "NB": 300,
        "NM": 1000,
        "arr_rate_mean": 12,
        "mean_bid_val": 12,
        "var": 1.8,
        "mean_duration": 30,
        "SLAV": 10,
        "rej_penalty": 0,
        "slav_penalty": 10,
    },
}

total_res = {"C": 100000, "B": 70000, "M": 100000}

class_types = ["C1", "C2", "C3"]
