# История экспериментов

По умолчанию ClearML закрыт, откра лучшая модель (см. [README](README.md))

## 29.04.2024, exp-1 Дмитрий

- https://app.clear.ml/projects/cb019a605a934ca1a4d85897c43bec3b/experiments/a366ecbfd8fd4ba6a19f0fecf4842884/artifacts/output-model/06cb12f34bc24d16a26767983b6018f4?columns=selected&columns=type&columns=name&columns=tags&columns=status&columns=project.name&columns=users&columns=started&columns=last_update&columns=last_iteration&columns=parent.name&order=-last_update&filter=
- модель Resnet50+CRNN

### Что не зашло
- неверно определил метрики, не считались

### Идеи на будущее
- проверить код

---


## 29.04.2024, exp-1 Дмитрий

- https://app.clear.ml/projects/cb019a605a934ca1a4d85897c43bec3b/experiments/a118c3a4d0e449b4a1dd84358c0f1fa5/artifacts/output-model/6bfbce2af5ec45d8b55d59a714a5600b?columns=selected&columns=type&columns=name&columns=tags&columns=status&columns=project.name&columns=users&columns=started&columns=last_update&columns=last_iteration&columns=parent.name&order=-last_update&filter=
- модель Resnet50+CRNN
- общая метрика string_match = 0.87

### Что не зашло
- забыл включить сохранение чекпоинтов модели

### Идеи на будущее
- проверить код

---


## 29.04.2024, exp-1 Дмитрий

- https://app.clear.ml/projects/cb019a605a934ca1a4d85897c43bec3b/experiments/3b62d34fc96049ee9f82db6c858be152/output/execution
- модель Resnet50+CRNN
- общая метрика string_match = 0.94

### Что не зашло
- нет предсказания типа номера (какой страны/региона)

### Идеи на будущее
- понять как добавить классификационную голову, вывод текста региона 
  реализован в Dataset

---
