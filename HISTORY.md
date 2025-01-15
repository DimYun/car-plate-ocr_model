# History of experiments

ClearML is closed by default, except for best experiment logs (see [README](README.md))


## 29.04.2024, exp-1, Dmitrii

- https://app.clear.ml/projects/cb019a605a934ca1a4d85897c43bec3b/experiments/a366ecbfd8fd4ba6a19f0fecf4842884/artifacts/output-model/06cb12f34bc24d16a26767983b6018f4?columns=selected&columns=type&columns=name&columns=tags&columns=status&columns=project.name&columns=users&columns=started&columns=last_update&columns=last_iteration&columns=parent.name&order=-last_update&filter=
- Resnet50+CRNN model

### Good
-

### Bad
- wrong metric setup, can't calculate

### Ideas
- debug code

---


## 29.04.2024, exp-2, Dmitrii

- https://app.clear.ml/projects/cb019a605a934ca1a4d85897c43bec3b/experiments/a118c3a4d0e449b4a1dd84358c0f1fa5/artifacts/output-model/6bfbce2af5ec45d8b55d59a714a5600b?columns=selected&columns=type&columns=name&columns=tags&columns=status&columns=project.name&columns=users&columns=started&columns=last_update&columns=last_iteration&columns=parent.name&order=-last_update&filter=
- Resnet50+CRNN model
- total metric string_match = 0.87

### Good
- correct calculation

### Bad
- forgot to turk on save checkpoint

### Ideas
- debug code

---


## 29.04.2024, exp-3, Dmitrii

- https://app.clear.ml/projects/cb019a605a934ca1a4d85897c43bec3b/experiments/3b62d34fc96049ee9f82db6c858be152/output/execution
- Resnet50+CRNN model
- Total metric string_match = 0.94

### Good
- correct calculations

### Bad
- no plate number type classification (country or region)

### Ideas
- add classification

---
