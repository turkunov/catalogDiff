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
    assert "Qwen/Qwen3.5-9B" in services["vllm"]["command"] and "--max-model-len 16384" in services["vllm"]["command"]
    assert (ROOT / "docker" / "vllm-cu129" / "Dockerfile").exists() and (ROOT / ".env.example").exists()


def test_ablation_servers_pin_distinct_gpus_and_bind_localhost():
    services = yaml.safe_load((ROOT / "docker-compose.ablations.yml").read_text())["services"]
    gpus = [svc["gpus"][0]["device_ids"][0] for svc in services.values()]
    main_vllm_gpu = "7"
    assert len(gpus) == len(set(gpus)) == 7 and main_vllm_gpu not in gpus
    for name, svc in services.items():
        assert all(str(p).startswith("127.0.0.1:") for p in svc["ports"]), name
        assert "HF_HUB_DISABLE_XET=1" in svc["environment"]
    models = " ".join(svc["command"] for svc in services.values())
    for m in ("Qwen/Qwen3.5-9B", "Qwen/Qwen3.5-4B", "Qwen/Qwen2.5-VL-7B-Instruct", "OpenGVLab/InternVL3_5-8B"):
        assert m in models
    assert "--max-num-seqs 16" in compose()["services"]["vllm"]["command"]


def test_ablation_config_variants_point_to_ablation_servers():
    cfg = yaml.safe_load((ROOT / "experiments" / "ablations.yaml").read_text())
    services = set(yaml.safe_load((ROOT / "docker-compose.ablations.yml").read_text())["services"]) | {"vllm"}
    names = [v["name"] for v in cfg["variants"]]
    assert len(names) == len(set(names)) and "base" in names and cfg["threshold"] == 4
    for v in cfg["variants"]:
        assert v["base_url"].split("//")[1].split(":")[0] in services, v["name"]
