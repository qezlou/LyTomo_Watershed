import argparse

def download_files(ACCESS_TOKEN, file=None, datadir_path = '../LyTomo_data'):
    
    import requests
    import numpy as np
    import os
    
    if not os.path.exists(datadir_path):
        os.system('mkdir '+datadir_path)
    
    record_id = 5770883 # Our repository's record ID on zenodo.org

    r = requests.get(f"https://zenodo.org/api/records/{record_id}",
                     params={'access_token': ACCESS_TOKEN})
    
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
        r = requests.get(download_urls[ind], params={'access_token': ACCESS_TOKEN})
        with open(os.path.join(datadir_path, fname), 'wb') as f:
            f.write(r.content)
            
if __name__== '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', type=str, required=True, 
                        help='Your Access Toekn')
    parser.add_argument('-f', type=str, required=False, default=None,
                       help="Provide the files' names you want ro downalod")
    args = parser.parse_args()
    download_files(ACCESS_TOKEN=args.t, file= args.f)
