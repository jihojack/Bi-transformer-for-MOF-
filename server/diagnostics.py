"""Small, flushed startup records. Never pass paths, URLs, tokens or exceptions."""
import json
import os
from pathlib import Path
import time


def resource_snapshot():
    data={}
    try:
        pages=int(Path('/proc/self/statm').read_text().split()[1])
        data['rss_mib']=round(pages*os.sysconf('SC_PAGE_SIZE')/1024**2,1)
    except (OSError,ValueError,IndexError,AttributeError):
        pass
    try:
        import resource
        data['peak_rss_mib']=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,1)
    except (ImportError,OSError,ValueError):
        pass
    for name,filename in [('container_memory_mib','memory.current'),('container_memory_limit_mib','memory.max')]:
        try:
            value=Path('/sys/fs/cgroup',filename).read_text().strip()
            if value!='max':data[name]=round(int(value)/1024**2,1)
        except (OSError,ValueError):
            pass
    return data


def emit(event,**fields):
    # Only callers' explicit, sanitized scalar fields are accepted. Never log
    # exception messages/tracebacks: ONNX and HTTP exceptions can contain secrets.
    record={'event':event,'pid':os.getpid(),'utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),**fields,**resource_snapshot()}
    try:print('[mof_startup] '+json.dumps(record,ensure_ascii=True),flush=True)
    except (OSError,ValueError):pass
