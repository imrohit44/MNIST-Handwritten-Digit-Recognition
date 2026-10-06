import urllib.request, json
d = json.load(urllib.request.urlopen('https://pypi.org/pypi/tensorflow/2.21.0/json', timeout=30))
for u in d['urls']:
    if 'win_amd64' in u['url'] and ('cp313' in u['url'] or 'manylinux' in u['url']):
        print(u['url'])
        open('tf_url.txt','w').write(u['url'])
        break
else:
    print('No windows cp313 wheel found. All URLs:')
    for u in d['urls']:
        print(' ', u['url'])

