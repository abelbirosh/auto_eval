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
