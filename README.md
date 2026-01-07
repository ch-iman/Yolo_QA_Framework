# Yolo_QA_Framework

# Framework de Test QA Automatisé pour Modèles YOLO

<div align="center">
<img width="947" height="475" alt="report" src="https://github.com/user-attachments/assets/7229ff7c-15e1-4917-8a0f-b544b7996937" />

<img width="953" height="446" alt="image" src="https://github.com/user-attachments/assets/5e5b9ee4-6ca4-4cb1-a86e-1ad96e7343bd" />

**De la validation laboratoire à la certification industrielle**

</div>

---

## 🚨 Le Problème

Un modèle YOLO avec **95% de mAP en laboratoire** peut échouer en production. Pourquoi ? Parce qu'on teste la précision, jamais la robustesse.

**Ce framework valide automatiquement** :
- ✅ La fiabilité en production
- ✅ L'absence de régression après mise à jour
- ✅ La robustesse face aux conditions dégradées

---

## ✨ Fonctionnalités

### 4 Catégories de Tests (49 tests automatisés)

| 🔍 Fonctionnels | ⚡ Performance | 📈 Régression | 🌫️ Robustesse |
|----------------|---------------|--------------|---------------|
| Intégrité modèle | Latence CPU/GPU | Comparaison baseline | Flou, bruit |
| Format prédictions | Débit FPS | Détection auto | Luminosité |
| Classes valides | Utilisation mémoire | Alertes > 5% | Compression JPEG |
| Bounding boxes | Benchmarks | Sauvegarde auto | Résolutions |

### 📊 Résultats

| Condition | mAP | Δ% | Status |
|-----------|-----|-----|---------|
| Normale | 95.2% | 0% | ✅ |
| Flou (σ=3) | 87.4% | -8.2% | ✅ |
| Bruit (σ=25) | 91.1% | -4.3% | ✅ |
| JPEG Q=30 | 89.7% | -5.8% | ✅ |

---

## 🚀 Installation

```bash
git clone https://github.com/votre-username/yolo-qa-framework.git
cd yolo-qa-framework
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

---

## 💻 Utilisation

```bash
# Tous les tests
pytest

# Par catégorie
pytest -m functional    # Tests fonctionnels
pytest -m performance   # Tests de performance
pytest -m regression    # Tests de régression
pytest -m robustness    # Tests de robustesse

# Rapport HTML
pytest --html=reports/report.html
```

**Sortie** :
```
======================== 49 passed in 8.2s =========================
```

---

## 🏗️ Architecture 

```
<img width="1268" height="678" alt="image" src="https://github.com/user-attachments/assets/d21d96a2-1d71-470f-ba64-f6fde4fca8f9" />

```

---

## 🔄 CI/CD Pipeline

**6 jobs parallélisés** en < 8 minutes :
1. Tests fonctionnels
2. Tests performance
3. Tests régression
4. Tests robustesse
5. Génération rapports
6. Notificarion Slack

---

## 💡 Impact

### 🏭 Valeur Industrielle
- ✅ Réduction de **80%** du temps de validation manuelle
- ✅ Détection automatique des régressions
- ✅ Framework production-ready

### 🎓 Compétences Développées
- Software Engineering pour ML (Pytest, CI/CD)
- GitHub Actions (6 jobs parallélisés)
- Architecture extensible et reproductible

---

## 🎯 Roadmap

- [x] 49 tests automatisés ✅
- [x] Pipeline CI/CD complet ✅
- [ ] Extension à 50 images COCO
- [ ] Calcul IoU sur golden dataset
- [ ] Tests quantification (INT8/FP16)
- [ ] Déploiement NVIDIA Jetson Nano


<div align="center">

### ⭐ Si ce projet vous est utile, donnez-lui une étoile !


</div>
