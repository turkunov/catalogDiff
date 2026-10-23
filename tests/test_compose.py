import yaml

from helpers import ROOT


def compose():
    return yaml.safe_load((ROOT / "docker-compose.yml").read_text())


def test_ports_localhost_only():
    for name, svc in compose()["services"].items():
        for p in svc.get("ports", []):
            assert str(p).startswith("127.0.0.1:"), name


def test_grafana_password_from_env():
    env = compose()["services"]["grafana"]["environment"]
    password = [e for e in env if str(e).startswith("GF_SECURITY_ADMIN_PASSWORD=")][0]
    assert "${GRAFANA_ADMIN_PASSWORD" in password and "=admin" not in password


def test_services_and_models():
    services = compose()["services"]
    assert {"vllm", "app", "jupyter", "prometheus", "grafana"} <= set(services)
    assert "Qwen/Qwen3.5-27B" in services["vllm"]["command"] and "--max-model-len 8192" in services["vllm"]["command"]
    assert (ROOT / "docker" / "vllm-cu129" / "Dockerfile").exists() and (ROOT / ".env.example").exists()
