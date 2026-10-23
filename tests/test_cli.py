from pseudolabel.cli import build_parser, main


def test_cli_parses_commands():
    args = build_parser().parse_args(["export", "--profile", "sofas", "--threshold", "4", "--out", "/tmp/x"])
    assert args.cmd == "export" and args.threshold == 4
    args = build_parser().parse_args(["select-manual-test", "--profile", "sofas", "--pairs", "p.csv", "--from-csv", "set.csv"])
    assert args.from_csv == "set.csv"


def test_cli_export_without_threshold_fails(capsys, tmp_settings_env):
    assert main(["export", "--profile", "sofas", "--out", str(tmp_settings_env / "x")]) == 2
    assert "Positive class threshold is not set" in capsys.readouterr().err


def test_cli_select_and_import(capsys, tmp_settings_env, tmp_path):
    from helpers import write_small_csv

    pairs = write_small_csv(tmp_path)
    assert main(["select-manual-test", "--profile", "sofas", "--pairs", str(pairs), "--per-class", "1"]) == 0
    h = tmp_path / "h.csv"
    h.write_text("a_id,b_id,class\n2272175750151,380288750116,5\n")
    assert main(["import-human", "--profile", "sofas", "--csv", str(h)]) == 0
    assert "imported 1" in capsys.readouterr().out


def test_cli_labeler_uses_config_model(tmp_settings_env, sofa_profile):
    from pseudolabel.cli import _labeler
    from pseudolabel.settings import load_settings
    from pseudolabel.store import Store

    settings = load_settings()
    lab = _labeler(settings, Store(settings.db_path), sofa_profile, 1, model="other/model")
    assert lab.vlm.model == "other/model"
