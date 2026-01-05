"""
Générateur de Rapport QA - Version support skipped
==================================================

Génère un rapport HTML complet des tests YOLO, en prenant en compte
les tests passés, échoués et ignorés (skipped).

Usage:
    python scripts/generate_qa_report.py
"""

import json
from pathlib import Path
from datetime import datetime
import sys

# Fichier JSON généré par pytest
PYTEST_JSON_FILE = Path("reports/pytest_results.json")


def load_test_results():
    """Charge les résultats pytest depuis JSON"""
    if not PYTEST_JSON_FILE.exists():
        print(f"Aucun fichier pytest trouvé à {PYTEST_JSON_FILE}")
        sys.exit(1)
    
    with open(PYTEST_JSON_FILE) as f:
        data = json.load(f)

    # Initialisation des compteurs
    results_by_category = {
        "functional": {"passed": 0, "failed": 0, "skipped": 0},
        "performance": {"passed": 0, "failed": 0, "skipped": 0},
        "regression": {"passed": 0, "failed": 0, "skipped": 0}
    }

    # Parcourir tous les tests
    for test in data.get("tests", []):
        markers = test.get("nodeid", "")
        category = "functional" if "test_functional" in markers else \
                   "performance" if "test_performance" in markers else \
                   "regression" if "test_regression" in markers else None
        
        if category:
            outcome = test.get("outcome")
            if outcome == "passed":
                results_by_category[category]["passed"] += 1
            elif outcome == "skipped":
                results_by_category[category]["skipped"] += 1
            else:
                results_by_category[category]["failed"] += 1

    return results_by_category


def generate_html_report(results):
    """Génère le rapport HTML dynamique"""
    html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>YOLO QA Report</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background: #f5f5f5;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 30px;
        }}
        .header h1 {{
            margin: 0;
            font-size: 2.5em;
        }}
        .header p {{
            margin: 10px 0 0 0;
            opacity: 0.9;
        }}
        .section {{
            background: white;
            padding: 25px;
            margin-bottom: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .section h2 {{
            color: #667eea;
            margin-top: 0;
            border-bottom: 2px solid #667eea;
            padding-bottom: 10px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background: #667eea;
            color: white;
        }}
        tr:hover {{
            background: #f5f5f5;
        }}
        .status-pass {{
            color: #28a745;
            font-weight: bold;
        }}
        .status-fail {{
            color: #dc3545;
            font-weight: bold;
        }}
        .status-skipped {{
            color: #ffc107;
            font-weight: bold;
        }}
        .footer {{
            text-align: center;
            margin-top: 40px;
            padding: 20px;
            color: #666;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>YOLO QA Report</h1>
        <p>Rapport généré le {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>

    <div class="section">
        <h2>Résumé des Tests</h2>
        <table>
            <tr>
                <th>Catégorie</th>
                <th>Status</th>
                <th>Détails</th>
            </tr>
"""
    # Parcourir les catégories et afficher PASS/FAIL/SKIPPED
    for cat, vals in results.items():
        if vals["failed"] > 0:
            status = '<span class="status-fail">FAIL</span>'
        elif vals["skipped"] > 0:
            status = '<span class="status-skipped">SKIPPED</span>'
        else:
            status = '<span class="status-pass">PASS</span>'
        
        details = f"{vals['passed']} passed / {vals['failed']} failed / {vals['skipped']} skipped"
        html += f"""
            <tr>
                <td>{cat.capitalize()}</td>
                <td>{status}</td>
                <td>{details}</td>
            </tr>
"""
    html += """
        </table>
    </div>

    <div class="footer">
        <p>YOLO QA Framework | Généré automatiquement</p>
    </div>
</body>
</html>
"""
    return html


def main():
    results = load_test_results()
    html = generate_html_report(results)
    
    output_dir = Path("reports")
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / "qa_report.html"
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html)
    
    print(f"Rapport généré : {output_file}")


if __name__ == "__main__":
    main()
