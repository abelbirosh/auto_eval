from auto_eval.cohort import Endpoint
from auto_eval.endpoint import Hit, call, dig, parse_hits


class _Obj:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class FakeHTTP:
    """Records what went out and hands back what it was told to."""

    def __init__(self, payload=None, status=200, text="{}", raises=None):
        self.payload, self.status, self.body, self.raises = (
            payload,
            status,
            text,
            raises,
        )
        self.sent = []

    def _respond(self, **sent):
        self.sent.append(sent)
        if self.raises:
            raise self.raises
        return _Obj(status_code=self.status, text=self.body, json=lambda: self.payload)

    def get(self, url, params=None, headers=None, **kw):
        return self._respond(method="GET", url=url, params=params, headers=headers)

    def post(self, url, json=None, headers=None, **kw):
        return self._respond(method="POST", url=url, body=json, headers=headers)


def endpoint(**kwargs) -> Endpoint:
    base = dict(url="https://vendor.test/search", body={"query": "{query}"})
    base.update(kwargs)
    return Endpoint(**base)


# --- reading a vendor's answer --------------------------------------------


def test_results_are_read_through_the_declared_mapping():
    mapped = Endpoint(
        url="https://v.test/s",
        results_path="data.items",
        fields={"title": "name", "url": "link", "snippet": "body"},
    )
    hits = parse_hits(
        {"data": {"items": [{"name": "N", "link": "L", "body": "B"}]}}, mapped
    )
    assert hits == [Hit(rank=1, title="N", url="L", snippet="B")]


def test_a_response_that_is_a_bare_list_still_parses():
    hits = parse_hits(
        [{"title": "T"}], Endpoint(url="https://v.test/s", results_path="")
    )
    assert hits[0].title == "T" and hits[0].rank == 1


def test_a_shape_we_cannot_find_gives_no_hits_rather_than_an_error():
    assert parse_hits({"nothing": "here"}, endpoint()) == []


def test_dotted_paths_walk_dicts_and_lists():
    assert dig({"a": {"b": [10, 20]}}, "a.b.1") == 20
    assert dig({"a": 1}, "a.b") is None
    assert dig({"a": 1}, "") == {"a": 1}


# --- making the call ------------------------------------------------------


def test_only_the_query_goes_out(monkeypatch):
    """The gold answer never leaves this machine."""
    http = FakeHTTP(payload={"results": [{"title": "t", "url": "u", "text": "x"}]})
    response = call(
        endpoint(body={"query": "{query}", "numResults": 10}), "who?", client=http
    )
    assert response.ok and response.hits[0].title == "t"
    assert http.sent[0]["body"] == {"query": "who?", "numResults": 10}


def test_a_get_endpoint_sends_its_params(monkeypatch):
    http = FakeHTTP(payload={"results": []})
    call(endpoint(method="GET", params={"q": "{query}"}), "who?", client=http)
    assert http.sent[0] == {
        "method": "GET",
        "url": "https://vendor.test/search",
        "params": {"q": "who?"},
        "headers": {},
    }


def test_keys_are_read_from_the_environment_at_call_time(monkeypatch):
    monkeypatch.setenv("VENDOR_KEY", "sk-secret")
    http = FakeHTTP(payload={"results": []})
    call(endpoint(headers={"x-api-key": "${VENDOR_KEY}"}), "q", client=http)
    assert http.sent[0]["headers"]["x-api-key"] == "sk-secret"


def test_a_missing_key_is_an_error_not_a_call(monkeypatch):
    monkeypatch.delenv("VENDOR_KEY", raising=False)
    http = FakeHTTP(payload={"results": []})
    response = call(endpoint(headers={"x-api-key": "${VENDOR_KEY}"}), "q", client=http)
    assert not response.ok and "VENDOR_KEY" in response.error
    assert http.sent == []


def test_an_http_error_is_recorded_with_its_status():
    http = FakeHTTP(status=500, text="upstream is down")
    response = call(endpoint(), "q", client=http)
    assert (
        not response.ok
        and response.status == 500
        and "upstream is down" in response.error
    )


def test_a_non_json_response_is_an_error_not_a_crash():
    http = FakeHTTP(payload=None, text="<html>nope</html>")
    http.json = None

    class Broken(FakeHTTP):
        def _respond(self, **sent):
            self.sent.append(sent)

            def boom():
                raise ValueError("not json")

            return _Obj(status_code=200, text="<html>", json=boom)

    response = call(endpoint(), "q", client=Broken())
    assert not response.ok and "not JSON" in response.error


def test_a_timeout_is_recorded_against_that_item_only():
    response = call(endpoint(), "q", client=FakeHTTP(raises=TimeoutError("timed out")))
    assert not response.ok and "TimeoutError" in response.error


def test_an_address_this_machine_should_not_open_is_refused():
    """A cohort file pointing at the metadata endpoint is refused, not fetched."""
    response = call(Endpoint(url="http://169.254.169.254/latest/meta-data/"), "q")
    assert not response.ok and "private" in response.error.lower()


class FlakyHTTP:
    """Rate-limits the first `limits` calls, then answers."""

    def __init__(self, limits, payload=None, headers=None, status=429):
        self.limits, self.payload = limits, payload or {"results": []}
        self.headers, self.status = headers or {}, status
        self.calls = 0

    def _respond(self, **sent):
        self.calls += 1
        if self.calls <= self.limits:
            return _Obj(
                status_code=self.status,
                text="rate limited",
                headers=self.headers,
                json=lambda: {},
            )
        return _Obj(status_code=200, text="{}", headers={}, json=lambda: self.payload)

    def get(self, url, params=None, headers=None, **kw):
        return self._respond()

    def post(self, url, json=None, headers=None, **kw):
        return self._respond()


def test_a_rate_limit_is_waited_out_rather_than_scored_as_a_failure():
    http = FlakyHTTP(limits=2, payload={"results": [{"title": "t", "text": "found"}]})
    waited = []
    response = call(endpoint(), "q", client=http, sleep=waited.append)
    assert response.error is None
    assert response.retries == 2 and http.calls == 3
    assert [h.snippet for h in response.hits] == ["found"]


def test_the_backoff_doubles_between_attempts():
    http = FlakyHTTP(limits=2)
    waited = []
    call(endpoint(), "q", client=http, sleep=waited.append)
    assert waited == [2.0, 4.0]


def test_the_vendors_own_retry_after_is_preferred_to_a_guess():
    http = FlakyHTTP(limits=1, headers={"retry-after": "3"})
    waited = []
    call(endpoint(), "q", client=http, sleep=waited.append)
    assert waited == [3.0]


def test_a_retry_after_longer_than_a_board_will_wait_falls_back_to_the_backoff():
    http = FlakyHTTP(limits=1, headers={"retry-after": "600"})
    waited = []
    call(endpoint(), "q", client=http, sleep=waited.append)
    assert waited == [2.0]


def test_a_per_minute_quota_can_actually_be_waited_out():
    """The whole point: these vendors meter per minute, so the waits must span one."""
    http = FlakyHTTP(limits=99)
    waited = []
    call(endpoint(), "q", client=http, sleep=waited.append)
    assert sum(waited) >= 60.0, waited


def test_a_vendor_asking_for_most_of_a_minute_is_obeyed():
    http = FlakyHTTP(limits=1, headers={"retry-after": "45"})
    waited = []
    call(endpoint(), "q", client=http, sleep=waited.append)
    assert waited == [45.0]


def test_a_rate_limit_that_survives_every_attempt_is_an_error():
    http = FlakyHTTP(limits=99)
    response = call(endpoint(), "q", client=http, sleep=lambda s: None)
    assert response.error is not None and "429" in response.error
    assert response.retries == 5 and http.calls == 6


def test_a_plain_failure_is_not_retried():
    http = FlakyHTTP(limits=99, status=500)
    response = call(endpoint(), "q", client=http, sleep=lambda s: None)
    assert http.calls == 1 and response.retries == 0
    assert response.error.startswith("HTTP 500")


def test_the_wait_between_attempts_is_not_reported_as_the_vendor_being_slow():
    http = FlakyHTTP(limits=2)
    response = call(endpoint(), "q", client=http, sleep=lambda s: None)
    assert response.seconds < 1.0
