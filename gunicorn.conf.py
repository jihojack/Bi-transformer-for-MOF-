"""Start native inference only AFTER worker initialization, including --preload."""


def post_worker_init(worker):
    worker.wsgi.extensions['mof_start_model']()
