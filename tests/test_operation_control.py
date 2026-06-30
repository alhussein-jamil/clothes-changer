import pytest

from outfit_studio.ui.operation_control import (
    OperationCancelled,
    begin_operation,
    bind_session,
    check_cancelled,
    request_stop,
)


def test_stop_request_cancels_bound_session():
    begin_operation("session-a")
    bind_session("session-a")
    check_cancelled()
    request_stop("session-a")
    with pytest.raises(OperationCancelled):
        check_cancelled()


def test_stop_does_not_cancel_other_sessions():
    begin_operation("session-a")
    bind_session("session-a")
    request_stop("session-b")
    check_cancelled()


def test_begin_operation_clears_previous_stop():
    begin_operation("session-a")
    bind_session("session-a")
    request_stop("session-a")
    begin_operation("session-a")
    check_cancelled()
