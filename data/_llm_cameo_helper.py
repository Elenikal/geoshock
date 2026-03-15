# Auto-generated helper - called by event_detector via subprocess
# Forces UTF-8 at OS level so no codec errors reach the parent process.
import sys, os, json, re, unicodedata
os.environ['PYTHONIOENCODING'] = 'utf-8'
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.stderr.reconfigure(encoding='utf-8', errors='replace')

def clean(s):
    return unicodedata.normalize('NFKD', str(s)).encode('ascii', 'ignore').decode('ascii')

data   = json.loads(sys.stdin.buffer.read().decode('utf-8'))
key    = data['key']
model  = data['model']
system = clean(data['system'])
text   = clean(data['text'])

import urllib.request
payload = json.dumps({
    'model': model, 'max_tokens': 512,
    'system': system,
    'messages': [{'role': 'user', 'content': 'Code these headlines:\n\n' + text}],
}, ensure_ascii=True).encode('ascii')

req = urllib.request.Request(
    'https://api.anthropic.com/v1/messages',
    data=payload,
    headers={
        'x-api-key': key,
        'anthropic-version': '2023-06-01',
        'content-type': 'application/json',
    },
    method='POST',
)
try:
    with urllib.request.urlopen(req, timeout=30) as r:
        resp = json.loads(r.read().decode('utf-8'))
    raw = resp['content'][0]['text'].strip()
    raw = re.sub(r'```(?:json)?|```', '', raw).strip()
    result = json.loads(raw)
    result['llm_used'] = True
    print(json.dumps(result, ensure_ascii=True))
except Exception as e:
    err = str(e).encode('ascii', 'replace').decode('ascii')
    print(json.dumps({'error': err, 'llm_used': False, 'overall_severity': 0.0,
                      'gpr_z_estimate': 0.0, 'dominant_cameo_codes': [],
                      'key_events': [], 'summary': 'LLM error.', 
                      'de_escalation_signals': False}))
