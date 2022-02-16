def download(token, data_dir, file=None):
    """
    Download produced data files from Zenodo
    token : Your Access Toekn, get it from
            https://zenodo.org/account/settings/applications/tokens/new/
    data_dir : Provide the files' names you want to save data in
    file : Name of the particular file you want to downlaod, e.g. descendants.zip. If None,
           it downloads all the data
    """
    import requests
    import numpy as np
    import os
    
    record_id = 5770883 # Our repository's record ID on zenodo.org

    r = requests.get(f"https://zenodo.org/api/records/{record_id}",
                     params={'access_token': token})
    
    download_urls = np.array([f['links']['self'] for f in r.json()['files']])
    filenames = np.array([f['key'] for f in r.json()['files']])

    print(r.status_code)
    
    if file is None:
        file = filenames
    else :
        file= [file]
    for fname in file:
        ind = np.where(filenames==fname)[0][0]
        print("Downloading:", fname, ' from ', download_urls[ind])
        r = requests.get(download_urls[ind], params={'access_token': token})
        with open(os.path.join(data_dir, fname), 'wb') as f:
            f.write(r.content)
