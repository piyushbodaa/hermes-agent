"""A2A transport controls, exercised without opening sockets."""

import socket
import urllib.request
from email.message import Message
from unittest.mock import Mock

import httpx
import pytest

from plugins.platforms.a2a import security, tools
from plugins.platforms.a2a.adapter import A2ARequestHandler


def test_push_dials_a_validated_address_with_original_tls_identity(monkeypatch):
    addresses = [(socket.AF_INET, socket.SOCK_STREAM, 0, "", ("93.184.216.34", 443))]
    lookup = Mock(return_value=addresses)
    monkeypatch.setattr(socket, "getaddrinfo", lookup)
    calls = []

    def record(request):
        calls.append(request)
        return httpx.Response(302, headers={"Location": "https://other.example/"})

    real_client = httpx.Client
    monkeypatch.setattr(httpx, "Client", lambda **kwargs: real_client(transport=httpx.MockTransport(record), **kwargs))
    status = security.post_callback("https://peer.example:8443/hook", {}, {}, localhost_mode=False)
    assert status == 302
    assert lookup.call_count == len(calls) == 1
    assert calls[0].url.host == "93.184.216.34"
    assert calls[0].headers["Host"] == "peer.example:8443"
    assert calls[0].extensions["sni_hostname"] == "peer.example"


def test_private_dns_answer_refuses_push_before_delivery(monkeypatch):
    addresses = [(socket.AF_INET, socket.SOCK_STREAM, 0, "", (ip, 443)) for ip in ("93.184.216.34", "10.0.0.1")]
    monkeypatch.setattr(socket, "getaddrinfo", lambda *a, **kw: addresses)
    with pytest.raises(ValueError, match="internal"):
        security.post_callback("https://peer.example/hook", {}, {}, localhost_mode=False)


@pytest.mark.parametrize(("host", "origin", "allowed"), [
    ("localhost:9900", None, True),
    ("[::1]:9900", "http://[::1]:9900", True),
    ("localhost:9900", "https://example.org", False),
    ("example.org", None, False),
])
def test_local_listener_validates_browser_origin_and_host(host, origin, allowed):
    handler = A2ARequestHandler.__new__(A2ARequestHandler)
    handler.server = Mock()
    handler.adapter._security_context.localhost_only.return_value = True
    handler.headers = Message()
    handler.headers["Host"] = host
    if origin is not None:
        handler.headers["Origin"] = origin
    assert handler._local_request_allowed() is allowed


def test_invalid_body_length_is_rejected_before_reading():
    handler = A2ARequestHandler.__new__(A2ARequestHandler)
    handler.server = Mock()
    handler.adapter._security_context.localhost_only.return_value = False
    handler.client_address = ("127.0.0.1", 4000)
    handler.headers = {"Content-Length": "-1"}
    handler._error = Mock()
    handler.rfile = Mock()
    handler.do_POST()
    assert handler._error.call_args.args[0] == 400
    handler.rfile.read.assert_not_called()


def test_authenticated_redirects_are_origin_bound():
    handler = tools._PeerRedirectHandler()
    original = urllib.request.Request("https://peer.example/", headers={"Authorization": "Bearer fixture"})
    same = handler.redirect_request(original, None, 302, "", {}, "https://peer.example:443/next")
    assert same.get_header("Authorization") == "Bearer fixture"
    with pytest.raises(ValueError, match="another origin"):
        handler.redirect_request(original, None, 302, "", {}, "https://other.example/")


def test_card_cannot_reassign_a_configured_bearer(monkeypatch):
    monkeypatch.setattr(tools, "_fetch_card", lambda *a: {"url": "https://other.example/"})
    post = Mock()
    monkeypatch.setattr(tools, "_http_post_json", post)
    peer = {"url": "https://peer.example/", "auth": {"type": "bearer", "token": "fixture"}}
    with pytest.raises(ValueError, match="different RPC origin"):
        tools._send_task("peer", peer, "hello", "context")
    post.assert_not_called()
