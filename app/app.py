"""Compatibility launcher: all normal app inference now runs on the server."""
from pathlib import Path
import os
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from server.app import app

if __name__=='__main__':
    print('Server UI: http://127.0.0.1:'+os.getenv('PORT','7866'),flush=True)
    app.run(host=os.getenv('MOF_BIND','127.0.0.1'),port=int(os.getenv('PORT','7866')),debug=False)
