from BaseHTTPServer import BaseHTTPRequestHandler, HTTPServer
from urlparse import urlparse
import urllib
import json

import sys

import subprocess

LENGTH = 50

visualizer_subprocess = subprocess.Popen(
    ['/usr/users/abau/torch/install/bin/th', '/usr/users/abau/seq2seq-comparison/get-saliency-map.lua', '-model_list', '/usr/users/abau/seq2seq-comparison/model_list.txt', '-max_len', str(LENGTH)],
    stdin = subprocess.PIPE,
    stdout = subprocess.PIPE,
    stderr = sys.stdout,
    cwd = '/usr/users/abau/seq2seq-comparison/'
)

NETWORKS = [
    'en-es-0',
    'en-fr-0',
    'en-ar-0',
    'en-ru-0',
    'en-zh-0'
]

class VisualizationHandler(BaseHTTPRequestHandler):
    def _set_headers(self, content_type = 'text/html'):
        self.send_response(200)
        self.send_header('Content-Type', content_type)
        self.end_headers()

    def do_GET(self):
        url = urlparse(self.path)

        if url.path in ['', '/', '/index.html']:
            with open('gradient-visualization.html', 'r') as f:
                self._set_headers(content_type = 'text/html')
                self.wfile.write(f.read())

        elif url.path == '/visualize':
            query = url.query
            query_components = dict(qc.split('=') for qc in query.split('&') if '=' in qc)

            if 'network' in query_components and 'index' in query_components and 'sentence' in query_components:
                network_name = urllib.unquote(query_components['network'])
                network_index = urllib.unquote(query_components['index'])
                sentence = urllib.unquote(query_components['sentence'])

                if 'perturbation_size' in query_components:
                    perturbation_size = urllib.unquote(query_components['perturbation_size'])
                else:
                    perturbation_size = 1

                print(network_name)
                print(network_index)
                print(perturbation_size)
                print(sentence.split(' '))

                # Validate
                if 0 <= int(network_index) < 500 and network_name in NETWORKS and len(sentence.split(' ')) < LENGTH - 2:
                    visualizer_subprocess.stdin.write(network_name + '\n')
                    visualizer_subprocess.stdin.write(network_index + '\n')
                    visualizer_subprocess.stdin.write(perturbation_size + '\n')
                    visualizer_subprocess.stdin.write(sentence + '\n')

                    # Wait for a response to come
                    line = visualizer_subprocess.stdout.readline()
                else:
                    line = json.dumps({'Error': 'Invalid request.'})
            else:
                line = json.dumps({'Error': 'Invalid request; need arguments (network), (index), and (sentence)'})

            self._set_headers(content_type = 'application/json')
            self.wfile.write(line)

httpd = HTTPServer(
    ('', 8082),
    VisualizationHandler
)

print('Running server on 8082')

httpd.serve_forever()
