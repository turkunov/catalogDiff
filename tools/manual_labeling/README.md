# Ручная разметка 200 пар (ручной тест)

Слепая разметка фиксированного набора: старый класс и ответы VLM в браузер не отправляются. Каждая метка сразу пишется в SQLite (`OUT_DIR/manual_labels.sqlite`) и в журнал `manual_labels_audit.jsonl`.

```bash
# набор: 40 пар на каждый старый класс, по возможности одна пара на объявление Avito
python select_pairs.py --pairs combined_labeled_df_balanced_small.csv --out out/manual_test_pairs.csv

# на avi-gn-fsk44 (порт только на 127.0.0.1)
docker run -d --name manual-labeling --restart unless-stopped -p 127.0.0.1:7870:7870 -e HOST=0.0.0.0 \
  -v $PWD/manual_labeling:/ml -v $PWD/data/images_small:/data/images_small:ro \
  --entrypoint python repo-gradio:latest /ml/app/app.py

# доступ с ноутбука: ssh -N -L 7870:127.0.0.1:7870 avi-gn-fsk44.msk.avito.ru → http://localhost:7870
```

Выгрузка: `GET /export.csv` (`a_id,b_id,class,comment,updated_at`) → `python -m pseudolabel import-human --profile sofas --csv ...`.
