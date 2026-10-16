# Подключает фикстуры курса (impl, ref, task_info) и флаг --impl для ручного запуска:
#   pytest content/blocks/C/module-0-gpt2/C00-causal-mha --impl=reference
pytest_plugins = ["llmlab.pytest_plugin"]
