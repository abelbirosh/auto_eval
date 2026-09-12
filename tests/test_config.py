import auto_eval.config as config
from auto_eval.config import DEFAULT_MODEL, get_settings, load_env


def test_defaults_when_nothing_is_set(monkeypatch):
    monkeypatch.setattr(config, "_ENV_LOADED", True)  # don't read a real .env
    monkeypatch.delenv("AUTO_EVAL_MODEL", raising=False)
    monkeypatch.delenv("AUTO_EVAL_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    settings = get_settings()
    assert settings.model == DEFAULT_MODEL
    assert settings.base_url is None
    assert settings.has_key is False


def test_explicit_model_beats_the_environment(monkeypatch):
    monkeypatch.setattr(config, "_ENV_LOADED", True)
    monkeypatch.setenv("AUTO_EVAL_MODEL", "from-env")
    assert get_settings(model="explicit").model == "explicit"
    assert get_settings().model == "from-env"


def test_blank_key_does_not_count_as_a_key(monkeypatch):
    monkeypatch.setattr(config, "_ENV_LOADED", True)
    monkeypatch.setenv("OPENAI_API_KEY", "   ")
    assert get_settings().has_key is False


def test_dotenv_is_read_but_a_real_env_var_wins(tmp_path, monkeypatch):
    (tmp_path / ".env").write_text("OPENAI_API_KEY=from-file\nAUTO_EVAL_MODEL=model-from-file\n")
    monkeypatch.setattr(config, "_ENV_LOADED", False)
    monkeypatch.setenv("OPENAI_API_KEY", "from-shell")
    monkeypatch.delenv("AUTO_EVAL_MODEL", raising=False)

    found = load_env(tmp_path)

    assert found == tmp_path / ".env"
    settings = get_settings()
    assert settings.api_key == "from-shell"       # exported variable wins
    assert settings.model == "model-from-file"    # file fills what is unset


def test_dotenv_is_found_by_walking_up(tmp_path, monkeypatch):
    (tmp_path / ".env").write_text("AUTO_EVAL_MODEL=walked-up\n")
    nested = tmp_path / "a" / "b"
    nested.mkdir(parents=True)
    monkeypatch.setattr(config, "_ENV_LOADED", False)
    monkeypatch.delenv("AUTO_EVAL_MODEL", raising=False)

    assert load_env(nested) == tmp_path / ".env"
    assert get_settings().model == "walked-up"
