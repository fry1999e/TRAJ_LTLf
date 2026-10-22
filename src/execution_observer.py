"""Optional, task-local observation hooks. No policy is enabled by default.

Hooks may stop an execution, but cannot replace semantic artifacts. They are
scoped by ContextVar so concurrent tasks cannot observe each other's evidence.
"""
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps

_observer = ContextVar('traj_execution_observer', default=None)
_stage = ContextVar('traj_execution_stage', default='unscoped')


def current_observer():
    return _observer.get()


def current_stage():
    return _stage.get()


@contextmanager
def request_stage(name):
    """Set transport accounting stage without treating a raw JSON reply as a native artifact."""
    token = _stage.set(name)
    try:
        emit('working_request', stage=name)
        yield
    finally:
        _stage.reset(token)


@contextmanager
def observe_execution(observer):
    token = _observer.set(observer)
    try:
        yield observer
    finally:
        _observer.reset(token)


def emit(event, **payload):
    observer = current_observer()
    if observer is not None:
        observer.event(event, payload)


def observed_stage(name):
    def decorate(function):
        @wraps(function)
        def wrapped(*args, **kwargs):
            if current_observer() is None:
                return function(*args, **kwargs)
            token = _stage.set(name)
            try:
                emit('stage_enter', name=name)
                result = function(*args, **kwargs)
                emit('stage_result', name=name, result=result)
                return result
            except Exception as exc:
                emit('stage_failure', name=name, error_type=type(exc).__name__, message=str(exc)[:2000])
                raise
            finally:
                _stage.reset(token)
        return wrapped
    return decorate
