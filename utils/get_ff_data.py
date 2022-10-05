#!/usr/bin/env python3

def get_data(urls):
    import requests

    for name, url in urls.items():
        file_destination = 'data/{}'.format(name)
        res = requests.get(url)
        if res.status_code == 200:  # http 200 means success
            with open(file_destination, 'wb') as file_handle:  # wb means Write Binary
                file_handle.write(res.content)
        
    return True

def what_urls_to_get():
    urls = {
        "F-F_Research_Data_5_Factors_2x3_daily_CSV.zip": "http://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_5_Factors_2x3_daily_CSV.zip",
        "F-F_Momentum_Factor_daily_CSV.zip": "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Momentum_Factor_daily_CSV.zip",
        "F-F_ST_Reversal_Factor_daily_CSV.zip": "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_ST_Reversal_Factor_daily_CSV.zip",
        "F-F_LT_Reversal_Factor_daily_CSV.zip": "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_LT_Reversal_Factor_daily_CSV.zip",
    }
    return urls

def main():
    urls = what_urls_to_get()
    data = get_data(urls)
    return



if __name__ == "__main__":
    main()
    